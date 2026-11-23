export {};

declare global {
  interface Window {
    desktop?: {
      request: <T>(method: string, path: string, body?: unknown) => Promise<T>;
      dataset: <T>(action: string, input?: {
        offset?: number;
        anchorId?: number;
        payload?: Record<string, unknown>;
      }) => Promise<T>;
    };
  }
}
