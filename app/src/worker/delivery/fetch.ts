// Axis L: the network. Streams the pinned file from its URL.
//
// Contract
//   - openDownload(url, signal) resolves once the server has answered, with
//     the file's size as the server states it (Content-Length; null if it
//     does not say) and its bytes as an async iterable of chunks, in order.
//   - A failure is a FetchError naming its reason: "network" (no answer),
//     "not-found" (404), "needs-sign-in" (401, 403) or "http" (any other
//     status that is not a success, with the status). Never retried
//     silently; an abort rejects with the signal's own reason.
//
// Design
//   The whole-file download alone: Frizzle reads the file's front from its
//   own copy, so no range requests.

export type FetchReason = "network" | "not-found" | "needs-sign-in" | "http";

export class FetchError extends Error {
  readonly reason: FetchReason;
  constructor(reason: FetchReason, detail?: string) {
    super(detail === undefined ? reason : `${reason} (${detail})`);
    this.name = "FetchError";
    this.reason = reason;
  }
}

export interface Download {
  readonly totalSize: number | null;
  readonly chunks: AsyncIterable<Uint8Array>;
}

export type OpenDownload = (
  url: string,
  signal?: AbortSignal,
) => Promise<Download>;

export const openDownload: OpenDownload = async (url, signal) => {
  let response: Response;
  try {
    response = await fetch(url, signal === undefined ? {} : { signal });
  } catch (error) {
    if (signal?.aborted === true) throw signal.reason;
    void error;
    throw new FetchError("network", new URL(url).host);
  }
  if (response.status === 401 || response.status === 403) {
    throw new FetchError("needs-sign-in", `status ${response.status}`);
  }
  if (response.status === 404) throw new FetchError("not-found");
  if (!response.ok || response.body === null) {
    throw new FetchError("http", `status ${response.status}`);
  }
  const length = response.headers.get("Content-Length");
  return {
    totalSize: length === null ? null : Number(length),
    chunks: chunksOf(response.body),
  };
};

async function* chunksOf(
  body: ReadableStream<Uint8Array>,
): AsyncIterable<Uint8Array> {
  const reader = body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      yield value;
    }
  } finally {
    reader.releaseLock();
  }
}
