"""Axis V: the Site build scenario (tests/README.md, integration).

Contract locked (scripts/site.py, web/): from this checkout, `site.py build`
builds the engine, the app and the site, and the site it writes, under a
base URL with a path (https://example.test/frizzle/), holds:
  - exactly the pages Home, About and the 404 page, beside the app's own
    index at app/;
  - nothing it names missing: every link, image, srcset entry, stylesheet
    and script of those four pages, every url() in the stylesheets, and
    every `new URL("…", import.meta.url)` in the app's scripts (its worker,
    the engine's module), resolves to a file inside the output, under the
    base's path;
  - the engine's module the wasm-release build made, the same bytes, once;
  - on each of the three pages, a title, a description, the canonical URL
    of the page itself, and a share image that is a 1200 × 630 JPEG inside
    the output, with its alt text.

Only this scenario covers the three builds chained as they publish: the
app's base and Hugo's mounts together, under a base with a path. Bugs that
would still pass: the app failing to run once loaded (no browser test
exists, tests/README.md), a page reading wrongly, an external link that is
dead, a published address that is not served.

Needs Docker with the pinned emsdk image, Node and the app's packages, and
Hugo at its pin; a missing one fails, never skips. Run with
`python3 scripts/check.py site`, when web/, the app's build or site.py
change.
"""
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[3]
# scripts/site.py by its path: `import site` names Python's own module,
# loaded at startup. Its imports (app, tooling) are its neighbours.
sys.path.insert(0, str(ROOT / 'scripts'))
_spec = importlib.util.spec_from_file_location('frizzle_site', ROOT / 'scripts' / 'site.py')
site_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(site_build)

BASE = 'https://example.test/frizzle/'
PAGES = {'index.html', 'about/index.html', '404.html'}


class _References(HTMLParser):
    """Every address a page names, and its head's metadata."""

    def __init__(self):
        super().__init__()
        self.addresses = []
        self.meta = {}
        self.canonical = None
        self.title = ''
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for name in ('href', 'src'):
            if a.get(name):
                self.addresses.append(a[name])
        if a.get('srcset'):
            self.addresses += [part.split()[0] for part in a['srcset'].split(',')]
        if tag == 'meta' and a.get('content') is not None:
            self.meta[a.get('property') or a.get('name')] = a['content']
        if tag == 'link' and a.get('rel') == 'canonical':
            self.canonical = a.get('href')
        self._in_title = tag == 'title'

    def handle_endtag(self, tag):
        self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def jpeg_size(data):
    """(width, height) from a JPEG's first start-of-frame marker."""
    if data[:2] != b'\xff\xd8':
        raise ValueError('not a JPEG')
    at = 2
    while at < len(data):
        marker, length = data[at + 1], struct.unpack('>H', data[at + 2:at + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack('>HH', data[at + 5:at + 9])
            return width, height
        at += 2 + length
    raise ValueError('no frame in the JPEG')


class SiteBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temporary.name) / 'public'
        site_build.build(cls.out, BASE)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def url_of(self, path):
        return urljoin(BASE, path.relative_to(self.out).as_posix())

    def file_of(self, address, at):
        """The output file an address names from the page or script at `at`; None if it is outside the site."""
        url = urljoin(self.url_of(at), address)
        parts = urlsplit(url)
        if parts.scheme not in ('http', 'https') or parts.netloc != 'example.test':
            return None
        base_path = urlsplit(BASE).path
        self.assertTrue(parts.path.startswith(base_path), f'{address} in {at.name} leaves the base: {url}')
        target = self.out / parts.path[len(base_path):]
        return target / 'index.html' if parts.path.endswith('/') else target

    def read(self, page):
        parser = _References()
        parser.feed((self.out / page).read_text())
        return parser

    def test_the_pages_are_home_about_and_404_beside_the_app(self):
        pages = {p.relative_to(self.out).as_posix() for p in self.out.rglob('*.html')
                 if not p.relative_to(self.out).as_posix().startswith('app/')}
        self.assertEqual(pages, PAGES)
        self.assertTrue((self.out / 'app' / 'index.html').is_file())

    def test_nothing_named_is_missing(self):
        checked = 0
        for page in sorted(PAGES) + ['app/index.html']:
            for address in self.read(page).addresses:
                target = self.file_of(address, self.out / page)
                if target is not None:
                    self.assertTrue(target.is_file(), f'{page} names {address}, not in the output')
                    checked += 1
        for sheet in self.out.rglob('*.css'):
            for address in re.findall(r'url\(\s*["\']?([^"\')]+)', sheet.read_text()):
                if address.startswith('data:'):
                    continue
                target = self.file_of(address, sheet)
                self.assertTrue(target is not None and target.is_file(), f'{sheet.name} names {address}')
                checked += 1
        scripts = list((self.out / 'app').rglob('*.js'))
        named = 0
        for script in scripts:
            for address in re.findall(r'new URL\(\s*[`"\']([^`"\']+)[`"\']\s*,\s*import\.meta\.url', script.read_text()):
                target = self.file_of(address, script)
                self.assertTrue(target is not None and target.is_file(), f'{script.name} names {address}')
                named += 1
        # The worker and the engine's module at least.
        self.assertGreaterEqual(named, 2)
        self.assertGreater(checked, 20)

    def test_the_engine_module_is_the_one_built(self):
        modules = list((self.out / 'app').rglob('*.wasm'))
        self.assertEqual(len(modules), 1, modules)
        built = ROOT / 'build' / 'wasm-release' / 'web' / 'frizzle.wasm'
        self.assertEqual(modules[0].read_bytes(), built.read_bytes())

    def test_every_page_carries_its_metadata_and_share_image(self):
        for page in sorted(PAGES):
            with self.subTest(page=page):
                head = self.read(page)
                self.assertTrue(head.title.strip())
                self.assertTrue(head.meta.get('description'))
                expected = BASE if page == 'index.html' else self.url_of(self.out / page).removesuffix('index.html')
                self.assertEqual(head.canonical, expected)
                self.assertEqual(head.meta.get('og:url'), expected)
                image = self.file_of(head.meta['og:image'], self.out / page)
                self.assertIsNotNone(image)
                self.assertEqual(jpeg_size(image.read_bytes()), (1200, 630))
                self.assertEqual(head.meta.get('twitter:image'), head.meta['og:image'])
                self.assertTrue(head.meta.get('og:image:alt'))


if __name__ == '__main__':
    unittest.main()
