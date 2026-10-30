// Axis H (shape): numbers as people read them. Plain TypeScript.
//
// Contract
//   - megabytes(bytes): decimal megabytes (10^6 bytes), rounded to the
//     nearest whole one, grouped in en-US: 721,918,496 → "722 MB"; under
//     one megabyte but not zero, "<1 MB".
//   - signed(v, digits = 2): a neuron's value for reading: a sign always
//     (+ or the minus sign −), `digits` places after the point: 8 → "+8.00",
//     -0.0798 → "−0.08"; 0 and -0 → "0.00"; ±Infinity → "+∞", "−∞"; NaN →
//     "NaN".
//   - significant(v): a scale's end for the legend, two significant digits,
//     no trailing zeros: 97.39 → "97", 0.5999 → "0.6", 12.79 → "13", 1.4 →
//     "1.4"; 0 → "0".
//   - Locale-free: en-US, named, never the browser's.
//
// Design
//   - These are for reading only. Wherever a test or a person needs the
//     value exactly, the view puts it in a <data value> beside the
//     rounded text (the exact f32, as String writes it).

const WHOLE = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const MINUS = "−";

export function megabytes(bytes: number): string {
  if (bytes > 0 && bytes < 500_000) return "<1 MB";
  return `${WHOLE.format(bytes / 1_000_000)} MB`;
}

export function signed(v: number, digits = 2): string {
  if (Number.isNaN(v)) return "NaN";
  if (!Number.isFinite(v)) return v > 0 ? "+∞" : `${MINUS}∞`;
  const text = Math.abs(v).toFixed(digits);
  if (Number(text) === 0) return (0).toFixed(digits);
  return (v > 0 ? "+" : MINUS) + text;
}

export function significant(v: number): string {
  if (v === 0 || !Number.isFinite(v)) return String(v);
  return String(Number(Math.abs(v).toPrecision(2)) * Math.sign(v));
}
