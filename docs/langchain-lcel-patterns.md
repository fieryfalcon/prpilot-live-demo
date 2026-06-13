# LangChain & LCEL: Deep-Dive Answers

> A practical, in-depth reference covering five core LangChain Expression
> Language (LCEL) and LangChain runtime topics. Each section is written to
> stand on its own, with conceptual explanations, real-world scenarios, and
> runnable code. Examples target `langchain` 0.2/0.3-era APIs using the
> `Runnable` protocol.

---

## Table of Contents

1. [RunnableSequence vs. RunnableParallel](#1-runnablesequence-vs-runnableparallel)
2. [Fallback Models in LangChain](#2-fallback-models-in-langchain)
3. [Callback Handlers: Monitoring, Logging, Token Tracking & Observability](#3-callback-handlers-monitoring-logging-token-tracking--observability)
4. [Streaming LLM Responses to a UI](#4-streaming-llm-responses-to-a-ui)
5. [Multi-Retriever Workflows with Dynamic Routing](#5-multi-retriever-workflows-with-dynamic-routing)

---

## 1. RunnableSequence vs. RunnableParallel

### 1.1 The mental model

LangChain Expression Language (LCEL) is built around a single abstraction:
the **`Runnable`**. A `Runnable` is anything that exposes a common interface —
`invoke()`, `batch()`, `stream()`, and their async twins `ainvoke()`,
`abatch()`, `astream()`. Prompts, chat models, output parsers, retrievers,
and even plain Python functions (wrapped in `RunnableLambda`) are all
`Runnable`s. Because they share one interface, they can be composed like Lego
bricks.

There are two fundamental ways to compose `Runnable`s, and they map directly
to the two questions every data-flow graph must answer: *"what runs after
what?"* (sequence) and *"what runs at the same time?"* (parallel).

**`RunnableSequence`** wires components end-to-end. The **output** of step *N*
becomes the **input** of step *N+1*. It models a pipeline: data flows in one
direction, transformed at each stage. In LCEL you almost never construct it by
name — the pipe operator `|` builds it for you:

```python
chain = prompt | model | output_parser
# is exactly equivalent to
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(prompt, model, output_parser)
```

**`RunnableParallel`** runs several `Runnable`s **concurrently on the same
input** and collects their outputs into a single dictionary, keyed by the names
you give them. In LCEL a plain Python `dict` inside a chain is *automatically
coerced* into a `RunnableParallel`:

```python
from langchain_core.runnables import RunnableParallel
parallel = RunnableParallel(
    joke=joke_chain,
    poem=poem_chain,
)
# equivalent shorthand inside a larger chain:
parallel = {"joke": joke_chain, "poem": poem_chain}
```

The key distinction: **sequence is about *order* (one feeds the next);
parallel is about *fan-out* (one input, many independent branches, combined
back into a dict).**

### 1.2 How data flows through each

Consider an input value `x`.

- In a `RunnableSequence` `a | b | c`, the computation is `c(b(a(x)))`.
  The shapes must line up: `a`'s output type must be acceptable as `b`'s input
  type. A common bug is putting a chat model directly before a string-only
  function without an output parser, because the model emits an `AIMessage`,
  not a `str`.

- In a `RunnableParallel({"p": a, "q": b})`, the computation is
  `{"p": a(x), "q": b(x)}`. Every branch receives the **same** `x`. The
  branches do not see each other's output. The result is always a dict.

Crucially, the two compose with each other. The single most common LCEL idiom
is a parallel block feeding a sequence:

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

Here the leading dict is a `RunnableParallel`. The original user question is
sent **simultaneously** to the `retriever` (which fetches documents) and to
`RunnablePassthrough` (which echoes the question unchanged). Both results land
in a dict `{"context": ..., "question": ...}`, which then flows **sequentially**
into the prompt, model, and parser. This one snippet shows both primitives
cooperating: parallel for fan-out, sequence for the pipeline.

### 1.3 Concurrency semantics that matter in production

`RunnableParallel` does not merely *describe* parallelism — it *executes* it.
When you call `invoke()`, the branches run on a thread pool; when you call
`ainvoke()`, they run as concurrent `asyncio` tasks. For I/O-bound branches —
which LLM and retriever calls almost always are — this means the **wall-clock
time of the parallel block is roughly the time of its slowest branch, not the
sum of all branches.** Two model calls that each take 2 seconds finish in ~2
seconds in parallel, versus ~4 seconds if you had chained them sequentially
for no reason.

That is the performance argument for `RunnableParallel`, but it has a second,
equally important role: **shaping data**. Even when branches are cheap, the
parallel dict is how you assemble the multi-field input that a downstream
prompt template expects.

### 1.4 Real-world scenario for RunnableSequence

**Scenario: a customer-support ticket summarizer-and-classifier pipeline.**

A SaaS company ingests raw support emails. For each email they want to (a)
clean and normalize the text, (b) summarize it into two sentences, (c) extract
a structured JSON object `{category, urgency, sentiment}`, and (d) validate
that JSON against a schema before writing it to the ticketing database.

Every step **depends on the previous one's output**: you cannot classify
before you have summarized; you cannot validate before you have JSON. This is
inherently sequential.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)

normalize = RunnableLambda(lambda text: text.strip().replace("\r\n", "\n"))

classify_prompt = ChatPromptTemplate.from_template(
    "Classify this support email. Return JSON with keys "
    "category, urgency (low/med/high), sentiment.\n\nEmail:\n{email}"
)

def validate(parsed: dict) -> dict:
    assert parsed["urgency"] in {"low", "med", "high"}, "bad urgency"
    return parsed

ticket_chain = (
    normalize
    | (lambda clean: {"email": clean})
    | classify_prompt
    | model
    | JsonOutputParser()
    | RunnableLambda(validate)
)

result = ticket_chain.invoke("  Hi, my invoice is wrong AGAIN!!! \r\n")
```

Each stage transforms and hands off. If any stage fails, the pipeline stops —
which is exactly what you want when later steps are meaningless without earlier
ones.

### 1.5 Real-world scenario for RunnableParallel

**Scenario: a "content studio" that generates a marketing bundle from one
product brief.**

A marketing tool takes a single product description and must produce, in one
request: a tweet, an SEO meta-description, three email subject lines, and a set
of hashtags. None of these outputs depends on the others — they are four
independent transformations of the **same** input. Running them sequentially
would quadruple the latency for no benefit.

```python
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

def make(instruction):
    p = ChatPromptTemplate.from_template(instruction + "\n\nProduct:\n{brief}")
    return p | model | StrOutputParser()

studio = RunnableParallel(
    tweet=make("Write a punchy <280-char tweet for this product."),
    meta=make("Write a 155-char SEO meta description."),
    subjects=make("Write 3 email subject lines, one per line."),
    hashtags=make("List 8 relevant hashtags, space separated."),
)

bundle = studio.invoke({"brief": "A solar-powered backpack charger..."})
# bundle == {"tweet": "...", "meta": "...", "subjects": "...", "hashtags": "..."}
```

All four model calls fire concurrently. The user waits for the slowest single
generation, not the total. This is the canonical fan-out use case.

### 1.6 Choosing between them — a decision checklist

| Question | If yes → use |
|---|---|
| Does step B need the *output* of step A? | `RunnableSequence` (`\|`) |
| Are the operations independent transforms of one input? | `RunnableParallel` (dict) |
| Do you need to build a multi-key input for a prompt? | `RunnableParallel` |
| Do you want to cut latency on independent I/O calls? | `RunnableParallel` |
| Is there a strict data dependency / validation order? | `RunnableSequence` |

In practice you will almost always use both together: parallel blocks to gather
and shape inputs, sequences to drive the resulting data through prompt → model
→ parser stages. Understanding which composition models *order* and which models
*fan-out* is the foundation of every non-trivial LCEL chain.

---

## 2. Fallback Models in LangChain

### 2.1 Why fallbacks exist

Production LLM applications fail in ways that ordinary web services do not.
A model endpoint can return a `429 Rate Limit`, time out under load, return a
`503` during a provider incident, hit a context-length error, or produce
malformed output that breaks a downstream parser. If your entire user
experience hangs off one model from one provider, every one of those events is
a full outage.

A **fallback** is a declarative answer to the question: *"if my primary path
fails, what should I try instead?"* LangChain bakes this directly into the
`Runnable` interface via the **`.with_fallbacks()`** method. Because *every*
`Runnable` supports it — not just chat models — you can attach fallbacks to a
single model, to a prompt-model pair, or to an entire RAG chain.

### 2.2 The core mechanism: `.with_fallbacks()`

`.with_fallbacks()` wraps a primary `Runnable` and a list of alternative
`Runnable`s into a `RunnableWithFallbacks`. At call time it tries the primary;
if the primary raises an exception, it tries the first fallback; if that raises,
it tries the next; and so on. The **first success wins**, and if everything
fails, the **last exception propagates**.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary = ChatOpenAI(model="gpt-4o", timeout=20, max_retries=2)
secondary = ChatAnthropic(model="claude-sonnet-4-6", timeout=20)
tertiary = ChatOpenAI(model="gpt-4o-mini")  # cheap, fast, last resort

model_with_fallbacks = primary.with_fallbacks([secondary, tertiary])

# If gpt-4o errors, Claude is tried; if that errors, gpt-4o-mini is tried.
answer = model_with_fallbacks.invoke("Explain LCEL fallbacks in one sentence.")
```

This is the direct answer to *"if GPT-4 fails, how does the workflow switch to
another model?"* — you give the GPT-4 runnable a list of alternatives and
LangChain handles the try/except chain for you. No manual exception handling, no
imperative `if-else` ladders.

### 2.3 Retries vs. fallbacks — two different layers

These are complementary, not interchangeable:

- **`max_retries`** (a model-level parameter) retries the **same** model on
  transient errors with exponential backoff. Good for a fleeting `429` or a
  network blip.
- **`.with_fallbacks()`** switches to a **different** runnable. Good for a
  sustained outage, a model-specific failure (e.g. context too long for model
  A but fine for model B), or graceful degradation to a cheaper model.

A robust setup uses both: each model retries itself a couple of times, and the
chain falls back to a different model if retries are exhausted.

```python
primary = ChatOpenAI(model="gpt-4o", max_retries=3)      # self-heals blips
backup  = ChatAnthropic(model="claude-sonnet-4-6", max_retries=2)
resilient = primary.with_fallbacks([backup])
```

### 2.4 Controlling *which* exceptions trigger a fallback

By default any `Exception` triggers the next fallback, but you usually want to
be selective — for example, fall back on rate limits and timeouts, but *not* on
an authentication error (which a fallback won't fix) or a content-policy refusal
(which you'd rather surface). Use `exceptions_to_handle`:

```python
from openai import RateLimitError, APITimeoutError

model_with_fallbacks = primary.with_fallbacks(
    fallbacks=[secondary],
    exceptions_to_handle=(RateLimitError, APITimeoutError),
)
```

Now an auth error from the primary propagates immediately instead of silently
masking a misconfiguration behind the backup model.

### 2.5 Falling back at the *chain* level, not just the model

A subtle but important point: different models often need **different prompts**.
GPT-4 might expect one system message format; a smaller open-weights model might
need a more explicit, few-shot prompt to perform acceptably. If you only fall
back the model, the backup inherits a prompt that wasn't tuned for it.

The fix is to attach fallbacks to entire `prompt | model | parser` chains, so
each path is internally consistent:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

gpt_chain = (
    ChatPromptTemplate.from_template("You are concise.\n\n{q}")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

# A more heavily-instructed prompt tuned for a smaller fallback model.
mini_chain = (
    ChatPromptTemplate.from_template(
        "Answer briefly and factually. If unsure, say so.\n\nQuestion: {q}\nAnswer:"
    )
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

robust_chain = gpt_chain.with_fallbacks([mini_chain])
robust_chain.invoke({"q": "What is the capital of Australia?"})
```

Each branch carries its own prompt, model, and parser, so a fallback is a
genuine drop-in replacement rather than a half-configured one.

### 2.6 Handling structured-output and parser failures

Fallbacks aren't only for network errors. A frequent real-world failure is the
model returning text that **breaks a parser** — e.g. invalid JSON. You can make
the *parser* robust two ways:

1. Use `OutputFixingParser`, which on a parse error sends the bad output back to
   an LLM with instructions to repair it.
2. Fall back from a strict, expensive extraction chain to a simpler one.

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser

base_parser = JsonOutputParser()
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=model)

extract = prompt | model | fixing_parser  # auto-repairs malformed JSON
```

### 2.7 A complete, production-shaped example

This combines retries, exception filtering, model fallbacks, and a final
"cheap and dumb but always-available" tier. The ordering encodes a cost/quality
gradient: try the best model, then a comparable model from another provider
(provider-diversity protects against single-vendor outages), then a small cheap
model as a guaranteed-something response.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import RateLimitError, APITimeoutError, APIError

prompt = ChatPromptTemplate.from_template("Answer helpfully:\n\n{question}")
parse = StrOutputParser()

tier1 = prompt | ChatOpenAI(model="gpt-4o", max_retries=3, timeout=30) | parse
tier2 = prompt | ChatAnthropic(model="claude-sonnet-4-6", max_retries=2) | parse
tier3 = prompt | ChatOpenAI(model="gpt-4o-mini", max_retries=1) | parse

answer_chain = tier1.with_fallbacks(
    fallbacks=[tier2, tier3],
    exceptions_to_handle=(RateLimitError, APITimeoutError, APIError),
)

print(answer_chain.invoke({"question": "Summarize the theory of relativity."}))
```

### 2.8 Operational guidance

- **Diversify providers.** A fallback to a second model from the *same* provider
  won't help during a provider-wide incident. Cross-provider fallbacks (OpenAI →
  Anthropic, or a self-hosted model) give true redundancy.
- **Log when a fallback fires.** Pair fallbacks with callbacks (Section 3) so you
  can alert on elevated fallback rates — a spike means your primary is degraded.
- **Mind cost and quality drift.** Falling back to a weaker model keeps you up
  but may lower answer quality; track which tier served each request.
- **Keep fallbacks fast.** Set tight timeouts on the primary so a hung call
  doesn't make users wait the full timeout *before* the fallback even starts.

Fallbacks turn a brittle single-model dependency into a resilient, tiered system
that degrades gracefully instead of failing outright.

---

## 3. Callback Handlers: Monitoring, Logging, Token Tracking & Observability

### 3.1 What a callback handler is

LangChain executes chains as a **tree of runs**: a top-level chain run contains
child runs for each prompt, model, retriever, tool, and parser it invokes. A
**callback handler** is an object that LangChain notifies as that tree executes.
It receives **lifecycle events** — "a chain started," "an LLM produced a new
token," "a tool finished," "a run errored" — and can react to each one.

This is the backbone of **observability** in LangChain. Instead of sprinkling
`print()` statements through your code, you attach a handler once and get a
structured, real-time stream of everything the runtime does.

A handler subclasses `BaseCallbackHandler` and overrides the event methods it
cares about. The most useful ones:

| Method | Fires when… |
|---|---|
| `on_chain_start` / `on_chain_end` | a chain/runnable begins / completes |
| `on_llm_start` / `on_chat_model_start` | a model call begins |
| `on_llm_new_token` | a token is streamed from the model |
| `on_llm_end` | a model call completes (carries token usage) |
| `on_tool_start` / `on_tool_end` | a tool/function call runs |
| `on_retriever_start` / `on_retriever_end` | a retriever runs |
| `on_*_error` | any of the above raises |

### 3.2 Attaching handlers: constructor vs. request scope

There are two ways to register a handler, and the difference matters:

```python
from langchain_core.callbacks import BaseCallbackHandler

# (a) Constructor callbacks — scoped to ONE component, for its whole life:
model = ChatOpenAI(model="gpt-4o", callbacks=[MyHandler()])

# (b) Request callbacks — scoped to ONE invocation, propagate to ALL children:
chain.invoke(input, config={"callbacks": [MyHandler()]})
```

Form **(b)** is almost always what you want for observability: passing handlers
through the `config` means they automatically propagate to every nested runnable
in the call tree, so one handler sees the prompt, the model, the retriever, and
the parser without being attached to each individually.

### 3.3 A custom handler for logging and monitoring

Here is a handler that logs the structure and timing of a run — the kind of
thing you'd wire into your application logs or a metrics backend like
Prometheus/StatsD.

```python
import time, logging
from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger("llm")

class MonitoringHandler(BaseCallbackHandler):
    def __init__(self):
        self._starts = {}

    def on_llm_start(self, serialized, prompts, *, run_id, **kw):
        self._starts[run_id] = time.perf_counter()
        log.info("LLM start run=%s prompt_chars=%d", run_id, len(prompts[0]))

    def on_llm_end(self, response, *, run_id, **kw):
        dt = time.perf_counter() - self._starts.pop(run_id, time.perf_counter())
        usage = (response.llm_output or {}).get("token_usage", {})
        log.info("LLM end run=%s latency=%.2fs tokens=%s", run_id, dt, usage)

    def on_llm_error(self, error, *, run_id, **kw):
        log.error("LLM error run=%s err=%s", run_id, error)

    def on_retriever_end(self, documents, *, run_id, **kw):
        log.info("Retriever returned %d docs run=%s", len(documents), run_id)

chain.invoke({"question": "..."}, config={"callbacks": [MonitoringHandler()]})
```

This single handler gives you per-model latency, error rates, and retrieval
counts — the raw material for dashboards and alerts.

### 3.4 Token tracking and cost accounting

Token usage drives cost, so tracking it is one of the most common observability
needs. There are several approaches:

**(a) The built-in OpenAI cost callback.** For OpenAI models, LangChain ships a
context manager that aggregates token counts and dollar cost across every model
call inside the block:

```python
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"question": "Write a 200-word essay on tea."})

print(f"Prompt tokens:     {cb.prompt_tokens}")
print(f"Completion tokens: {cb.completion_tokens}")
print(f"Total tokens:      {cb.total_tokens}")
print(f"Estimated cost:    ${cb.total_cost:.4f}")
```

**(b) `usage_metadata` on the response.** Modern chat models attach a
provider-agnostic `usage_metadata` dict (`input_tokens`, `output_tokens`,
`total_tokens`) to the `AIMessage`. Reading it in `on_llm_end` lets you build a
provider-neutral token tracker that also works for Anthropic, etc.

```python
class TokenTracker(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = self.output_tokens = 0

    def on_llm_end(self, response, **kw):
        for gen_list in response.generations:
            for gen in gen_list:
                meta = getattr(gen.message, "usage_metadata", None) or {}
                self.input_tokens += meta.get("input_tokens", 0)
                self.output_tokens += meta.get("output_tokens", 0)
```

This is the basis for **per-user or per-tenant billing**: instantiate a tracker
per request, tag it with the user ID, and write the totals to your metering
system.

### 3.5 Streaming token callbacks

`on_llm_new_token` fires for each token as it is generated (when streaming is
enabled). This is how you push partial output to a UI or measure
**time-to-first-token (TTFT)**, a key UX metric:

```python
class TTFTHandler(BaseCallbackHandler):
    def on_llm_start(self, *a, run_id, **kw):
        self._t0 = time.perf_counter()
        self._first = True

    def on_llm_new_token(self, token, **kw):
        if self._first:
            log.info("TTFT=%.3fs", time.perf_counter() - self._t0)
            self._first = False
```

Section 4 builds on this for end-to-end streaming to a frontend.

### 3.6 Async handlers

In an `asyncio` server (FastAPI, etc.), use `AsyncCallbackHandler` and override
the `async def` variants (`on_llm_new_token`, `on_llm_end`, …). This lets a
handler `await` I/O — for example, pushing tokens onto an `asyncio.Queue` or a
WebSocket without blocking the event loop:

```python
from langchain_core.callbacks import AsyncCallbackHandler

class QueueHandler(AsyncCallbackHandler):
    def __init__(self, queue): self.queue = queue
    async def on_llm_new_token(self, token, **kw):
        await self.queue.put(token)
    async def on_llm_end(self, response, **kw):
        await self.queue.put(None)  # sentinel: stream finished
```

### 3.7 Production observability: LangSmith and OpenTelemetry

For real deployments you rarely hand-roll everything. LangChain integrates with
**LangSmith**, which is essentially a hosted callback handler: set the
`LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` environment variables and
every run is traced automatically — full prompt/response capture, token usage,
latency, the run tree, and error stacks, viewable in a web UI with no code
changes.

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="ls-..."
export LANGCHAIN_PROJECT="support-bot-prod"
```

You can combine this with a custom handler that **also** exports spans to
OpenTelemetry, so LLM calls show up in the same distributed traces (Jaeger,
Datadog, Honeycomb) as the rest of your microservices. The general pattern:

- **LangSmith** for LLM-specific debugging (prompt inspection, eval, replay).
- **Custom handler → OTel/Prometheus** for infra-level SLOs (latency, error
  rate, cost per request) alongside the rest of your stack.

### 3.8 What callbacks give you, summarized

Callback handlers convert an opaque chain into an instrumented system. With
them you get: structured **logging** of every step, **monitoring** of latency
and error rates, **token tracking** for cost control and billing, **streaming**
hooks for responsive UIs, and **observability** that plugs into LangSmith,
OpenTelemetry, or your metrics backend — all without polluting business logic,
because the instrumentation lives in handlers attached via `config`.

---

## 4. Streaming LLM Responses to a UI

### 4.1 Why streaming matters

Large models generate text **token by token**. A 400-word answer might take 8–12
seconds to produce in full. If you wait for the complete response before showing
anything, the user stares at a spinner for the entire duration — and perceives
the app as slow even though the model is working the whole time.

**Streaming** sends each token to the user the instant it is produced. The
practical effects on user experience are large:

- **Time-to-first-token (TTFT) becomes the perceived latency.** Users see words
  appear in well under a second, so the app *feels* fast regardless of total
  length. This is the single biggest perceived-performance win in LLM UX.
- **It sets expectations and holds attention.** Watching text stream in is the
  now-familiar ChatGPT experience; a frozen screen reads as a hang.
- **It enables early cancellation.** If the answer is going the wrong way, the
  user can stop generation immediately, saving compute and tokens.
- **It supports incremental rendering** of long content (markdown, code) and
  progressive downstream processing.

### 4.2 The LCEL streaming primitives

Every `Runnable` exposes `.stream()` and `.astream()`. They return an
**iterator/async-iterator of output chunks**. For a chat model the chunks are
`AIMessageChunk` objects that **add together** to form the full message; for a
chain ending in `StrOutputParser`, the chunks are plain string fragments — which
is exactly what you want to append to a UI.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Write a short story about {topic}.")
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

for chunk in chain.stream({"topic": "a lighthouse keeper"}):
    print(chunk, end="", flush=True)   # each chunk is a string fragment
```

A critical detail about LCEL streaming: **the whole chain streams as long as
every component can stream incrementally.** Prompts, chat models, and
`StrOutputParser` all stream. But a step that must see the *entire* input before
producing output — for example a function that parses complete JSON — will
**buffer** the stream at that point. So design streaming chains so the final
user-facing step is stream-friendly (typically `StrOutputParser`).

### 4.3 Async streaming for web servers

Web servers are async, so you'll use `.astream()`. Here is the core loop in a
**FastAPI** endpoint using Server-Sent Events (SSE) — the simplest transport for
one-way server→client token streaming:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()
chain = (
    ChatPromptTemplate.from_template("{question}")
    | ChatOpenAI(model="gpt-4o", streaming=True)
    | StrOutputParser()
)

@app.get("/chat")
async def chat(question: str):
    async def event_stream():
        async for chunk in chain.astream({"question": question}):
            # SSE frames are "data: <payload>\n\n"
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

On the browser side, `EventSource` consumes it with almost no code:

```javascript
const es = new EventSource(`/chat?question=${encodeURIComponent(q)}`);
es.onmessage = (e) => {
  if (e.data === "[DONE]") { es.close(); return; }
  outputEl.textContent += e.data;   // append each token fragment
};
```

For bidirectional needs (the user can interrupt, send follow-ups on the same
connection) you'd use **WebSockets** instead, pushing each chunk with
`await ws.send_text(chunk)` inside the same `astream` loop.

### 4.4 `astream_events`: streaming *more* than the final tokens

Plain `.astream()` gives you the final output chunks. But real chat UIs often
want to show **intermediate** state too: "Searching documents…", tool calls,
which retriever fired, and only *then* the streamed answer. The
**`astream_events()`** API emits a rich, typed event for every step of the run
tree — start/stream/end events for chains, models, tools, and retrievers — so
you can drive a sophisticated UI from one stream.

```python
async for event in chain.astream_events({"question": q}, version="v2"):
    kind = event["event"]
    if kind == "on_retriever_end":
        yield sse({"type": "status", "text": "Found sources"})
    elif kind == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        if token:
            yield sse({"type": "token", "text": token})
    elif kind == "on_chat_model_end":
        yield sse({"type": "done"})
```

This is how production assistants show a "thinking → searching → answering"
progression: filter `astream_events` by event type and node name, and translate
each into a UI signal. It's the most powerful streaming tool LCEL offers.

### 4.5 Streaming with callbacks (the handler approach)

An alternative to consuming the iterator directly is to use a streaming
**callback handler** (Section 3) that pushes tokens onto an `asyncio.Queue`,
which your response generator drains. This decouples "where tokens come from"
from "where they go," and is handy when the LLM call is buried deep inside an
agent rather than being the top-level runnable:

```python
import asyncio
from langchain_core.callbacks import AsyncCallbackHandler

class SSEQueueHandler(AsyncCallbackHandler):
    def __init__(self, q): self.q = q
    async def on_llm_new_token(self, token, **kw): await self.q.put(token)
    async def on_llm_end(self, *a, **kw): await self.q.put(None)

async def chat(question):
    q = asyncio.Queue()
    task = asyncio.create_task(
        agent.ainvoke({"input": question}, config={"callbacks": [SSEQueueHandler(q)]})
    )
    while (tok := await q.get()) is not None:
        yield f"data: {tok}\n\n"
    await task
```

### 4.6 Practical concerns

- **Flush and disable buffering.** Reverse proxies (nginx) may buffer SSE; set
  `X-Accel-Buffering: no` and ensure the framework flushes each chunk.
- **Handle disconnects.** If the user closes the tab, detect the cancelled
  request and stop generation (`astream` will raise on cancellation) to avoid
  paying for tokens nobody will read.
- **Render incrementally but safely.** When streaming markdown/HTML, buffer just
  enough to avoid rendering broken syntax mid-token.
- **Always send a completion sentinel** (`[DONE]`) so the client knows when to
  close the connection and re-enable the input box.
- **Measure TTFT and tokens/sec** via a callback so you can monitor the metric
  users actually feel.

Streaming is not a nice-to-have for conversational UIs — it is the difference
between an app that feels instant and one that feels broken. LCEL makes it
nearly free: swap `.invoke()` for `.stream()`/`.astream()`, keep the final step
stream-friendly, and pipe chunks to SSE or WebSockets.

---

## 5. Multi-Retriever Workflows with Dynamic Routing

### 5.1 The problem: one retriever is rarely enough

A retriever turns a query into relevant documents. But a real application often
spans **heterogeneous knowledge sources** with different shapes and best-fit
retrieval strategies:

- A **product-docs** vector store (semantic search over manuals).
- A **SQL/structured** source for precise numeric or filtered lookups.
- A **code/API** index where keyword (BM25) search beats embeddings.
- A **web-search** tool for fresh, real-time questions.
- **Per-customer** knowledge bases that must be isolated for security.

No single retriever is best for every query. "What's our refund policy?" wants
the policy KB; "How many orders did I place in May?" wants structured data;
"What's the latest CVE for openssl?" wants live web search. The design goal is a
workflow that **inspects the query and routes it to the most appropriate
retriever (or retrievers), then synthesizes an answer.**

### 5.2 Three architectural patterns

There are three broadly useful strategies, often combined:

1. **Semantic routing** — pick *one* retriever based on which the query is most
   similar to. Fast and cheap; no LLM call to decide.
2. **LLM-based routing** — ask a model to *classify* the query and choose a
   route. More flexible; handles nuance and multi-step intent.
3. **Ensemble / fan-out + re-rank** — query *several* retrievers in parallel and
   merge/re-rank results. Highest recall; best when you can't confidently pick
   one source.

### 5.3 Pattern 1 — Semantic routing (embedding similarity)

Embed a short description (or example queries) of each retriever's domain. At
request time, embed the user query and route to the retriever whose description
is most similar. LangChain's utility makes this concise:

```python
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity

emb = OpenAIEmbeddings()

route_descriptions = {
    "billing":   "questions about invoices, payments, refunds, pricing",
    "technical": "API usage, errors, SDKs, integration, configuration",
    "account":   "login, passwords, profile, permissions, security",
}
route_names = list(route_descriptions)
route_vecs = emb.embed_documents(list(route_descriptions.values()))

retrievers = {
    "billing":   billing_vectorstore.as_retriever(),
    "technical": tech_vectorstore.as_retriever(),
    "account":   account_vectorstore.as_retriever(),
}

def pick_retriever(query: str):
    qv = emb.embed_query(query)
    sims = cosine_similarity([qv], route_vecs)[0]
    best = route_names[sims.argmax()]
    return retrievers[best]

router = RunnableLambda(lambda q: pick_retriever(q).invoke(q))
docs = router.invoke("Why was I charged twice last month?")  # → billing
```

This adds essentially no latency (one embedding call) and no per-decision LLM
cost. It's ideal when routes are well-separated by topic.

### 5.4 Pattern 2 — LLM-based routing with `RunnableBranch`

When routing needs reasoning — distinguishing "I can't log in" (account) from "I
can't log in *to the billing portal to dispute a charge*" (billing) — let a
model classify, then branch. First, a small classification chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

classifier = (
    ChatPromptTemplate.from_template(
        "Classify the query into exactly one of: billing, technical, account.\n"
        "Respond with only the label.\n\nQuery: {query}"
    )
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)
```

Then a `RunnableBranch` selects the retriever path based on the label. A
`RunnableBranch` is an LCEL `if/elif/else`: it evaluates conditions in order and
runs the first matching branch (the last argument is the default):

```python
def as_path(retriever):
    return (lambda x: x["query"]) | retriever

branch = RunnableBranch(
    (lambda x: x["route"] == "billing",   as_path(retrievers["billing"])),
    (lambda x: x["route"] == "technical", as_path(retrievers["technical"])),
    as_path(retrievers["account"]),  # default
)

route_and_retrieve = (
    {"query": RunnablePassthrough(), "route": classifier}
    | branch
)

docs = route_and_retrieve.invoke("I can't log in to my account")  # → account
```

Note the LCEL composition: a `RunnableParallel` computes the route and passes
the original query through simultaneously, then `RunnableBranch` dispatches.

A cleaner variant uses **structured output** so the classifier can't return an
unexpected label:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Route(BaseModel):
    destination: Literal["billing", "technical", "account"] = Field(
        description="Which knowledge base best answers the query"
    )

structured_classifier = (
    ChatPromptTemplate.from_template("Route this query: {query}")
    | ChatOpenAI(model="gpt-4o-mini").with_structured_output(Route)
)
```

### 5.5 Pattern 3 — Ensemble retrieval and re-ranking

Sometimes the right answer is *don't choose* — query multiple retrievers and
merge. LangChain's **`EnsembleRetriever`** does exactly this, combining results
from several retrievers using **Reciprocal Rank Fusion (RRF)**. The classic
combination is a sparse keyword retriever (BM25) with a dense vector retriever —
they have complementary strengths (exact-term match vs. semantic similarity),
so fusing them usually beats either alone (this is "hybrid search").

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs); bm25.k = 5
dense = vectorstore.as_retriever(search_kwargs={"k": 5})

hybrid = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.4, 0.6],   # tilt toward semantic, keep keyword recall
)
results = hybrid.invoke("ConnectionTimeoutError on the v2 /charges endpoint")
```

You can then **re-rank** the fused list with a cross-encoder for precision:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"), top_n=4
)
precise = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=hybrid
)
```

### 5.6 Putting it together: a dynamic, self-selecting RAG workflow

A production system layers these patterns: cheaply route most queries, fan-out
when the route is ambiguous, and always re-rank before answering. Below, the
classifier returns either a specific KB **or** `"unsure"`; on `"unsure"` we fall
back to the hybrid ensemble across all sources.

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def select(route: str):
    if route in retrievers:
        return retrievers[route]
    return hybrid  # "unsure" → query everything and fuse

def retrieve(payload):
    return select(payload["route"]).invoke(payload["query"])

answer_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

workflow = (
    {"query": RunnablePassthrough(), "route": classifier}
    | RunnablePassthrough.assign(docs=RunnableLambda(retrieve))
    | {
        "context": lambda x: format_docs(x["docs"]),
        "question": lambda x: x["query"],
      }
    | answer_prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

print(workflow.invoke("How do I rotate my API key and update billing email?"))
```

This single chain: (1) classifies the query, (2) dynamically selects the best
retriever — or the ensemble when unsure, (3) formats the retrieved context, and
(4) generates a grounded answer. Each stage is a `Runnable`, so the whole thing
inherits streaming, batching, fallbacks, and callback observability for free.

### 5.7 Design and operational considerations

- **Make routing observable.** Log the chosen route per query (via a callback)
  so you can audit misroutes and measure routing accuracy over time.
- **Always have a default/fallback route** (the ensemble) so an unexpected query
  never lands nowhere.
- **Mind metadata filtering for security.** When per-tenant isolation matters,
  pass a tenant filter into `search_kwargs` so a retriever can only return that
  customer's documents — routing alone is not an access-control boundary.
- **Balance cost vs. accuracy.** LLM routing is more accurate but adds a model
  call; semantic routing is near-free. A common hybrid: semantic route first,
  escalate to LLM routing only on low-confidence similarity.
- **Cache routing decisions** for repeated/similar queries to cut latency.
- **Evaluate end-to-end.** Use LangSmith datasets to measure whether the chosen
  retriever actually improved answer quality, not just whether routing "looks"
  right.

A well-designed multi-retriever workflow treats retrieval as a *routing problem*
layered over a *fusion problem*: route confidently when you can, fan-out and
re-rank when you can't, and instrument every decision so the system can be tuned
with data.

---

## Closing Notes

These five topics interlock. LCEL's `RunnableSequence`/`RunnableParallel`
composition is the substrate; **fallbacks** make any composed runnable
resilient; **callbacks** make it observable; **streaming** makes it feel
responsive; and **dynamic multi-retriever routing** is a sophisticated
application that uses all of the above. Mastering the `Runnable` interface —
one consistent API across prompts, models, retrievers, and functions — is what
lets these patterns combine so cleanly.
