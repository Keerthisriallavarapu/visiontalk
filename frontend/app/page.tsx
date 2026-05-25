"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, Send, Loader2, ImageIcon, Square } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8090";

export default function Home() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const onFile = useCallback((file: File) => {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setAnswer("");
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files?.[0];
      if (file && file.type.startsWith("image/")) onFile(file);
    },
    [onFile],
  );

  async function ask() {
    if (!imageFile || !question.trim()) return;
    setAnswer("");
    setError(null);
    setStreaming(true);

    const fd = new FormData();
    fd.append("image", imageFile);
    fd.append("question", question);
    fd.append("max_new_tokens", "300");
    fd.append("temperature", "0.2");

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const r = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        body: fd,
        signal: ctrl.signal,
      });
      if (!r.ok) throw new Error(await r.text());
      if (!r.body) throw new Error("no body");

      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        // SSE frames: separated by \n\n, each has "event: x\ndata: y"
        const frames = buf.split("\n\n");
        buf = frames.pop() ?? "";
        for (const frame of frames) {
          const lines = frame.split("\n");
          let event = "message";
          let data = "";
          for (const line of lines) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (event === "token" && data) setAnswer((a) => a + data);
          if (event === "done") {
            setStreaming(false);
            return;
          }
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") setError(e.message ?? "request failed");
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
    setStreaming(false);
  }

  return (
    <main className="min-h-screen bg-stone-50 text-stone-900">
      <div className="max-w-4xl mx-auto p-8">
        <header className="mb-6">
          <h1 className="text-3xl font-semibold tracking-tight flex items-center gap-2">
            <ImageIcon className="w-7 h-7 text-emerald-600" />
            VisionTalk
          </h1>
          <p className="text-stone-600 mt-1 text-sm">
            Drop an image, ask a question.
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div
            onDrop={onDrop}
            onDragOver={(e) => e.preventDefault()}
            className="bg-white border-2 border-dashed border-stone-300 rounded-lg p-4 flex flex-col items-center justify-center min-h-[280px] hover:border-emerald-400 transition"
          >
            {imagePreview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={imagePreview}
                alt="upload"
                className="max-h-72 max-w-full object-contain rounded"
              />
            ) : (
              <div className="text-center text-stone-500">
                <Upload className="w-10 h-10 mx-auto mb-2" />
                <p className="text-sm">Drop an image here</p>
              </div>
            )}
            <label className="mt-4 text-xs text-stone-500 hover:text-stone-800 cursor-pointer">
              or choose a file
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onFile(f);
                }}
              />
            </label>
          </div>

          <div className="flex flex-col gap-3">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What's in this image? What is the person doing?"
              rows={3}
              className="w-full border border-stone-300 rounded px-3 py-2 text-sm font-mono focus:ring-2 focus:ring-emerald-400 focus:outline-none"
            />
            <div className="flex gap-2">
              <button
                onClick={ask}
                disabled={!imageFile || !question.trim() || streaming}
                className="inline-flex items-center gap-2 bg-emerald-600 text-white px-4 py-2 rounded text-sm font-medium hover:bg-emerald-700 disabled:opacity-50"
              >
                {streaming ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                Ask
              </button>
              {streaming && (
                <button
                  onClick={stop}
                  className="inline-flex items-center gap-2 bg-stone-700 text-white px-3 py-2 rounded text-sm hover:bg-stone-800"
                >
                  <Square className="w-3.5 h-3.5" /> Stop
                </button>
              )}
            </div>

            <div className="bg-white border border-stone-200 rounded p-4 min-h-[200px]">
              {error && (
                <div className="text-red-700 text-sm mb-2">{error}</div>
              )}
              <div className="text-stone-800 whitespace-pre-wrap text-sm leading-relaxed font-mono">
                {answer || (
                  <span className="text-stone-400">
                    Answer will stream here...
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
