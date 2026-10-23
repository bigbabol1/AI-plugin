<p align="center">
  <img src="https://raw.githubusercontent.com/bigbabol1/AI-plugin/main/custom_components/ai_plugin/brand/icon@2x.png" alt="AI Plugin" width="128" height="128">
</p>

<h1 align="center">AI Plugin for Home Assistant</h1>

<p align="center">
  <b>A voice-first conversation agent for Home Assistant, built to run well on local LLMs and modest hardware.</b>
</p>

<p align="center">
  <a href="https://github.com/bigbabol1/AI-plugin/releases/latest"><img src="https://img.shields.io/github/v/release/bigbabol1/AI-plugin?style=flat-square&label=release" alt="Latest release"></a>
  <a href="https://hacs.xyz"><img src="https://img.shields.io/badge/HACS-custom-41BDF5?style=flat-square" alt="HACS custom repository"></a>
  <img src="https://img.shields.io/badge/Home%20Assistant-2025.7%2B-18BCF2?style=flat-square&logo=home-assistant&logoColor=white" alt="Home Assistant 2025.7 or newer">
  <a href="https://github.com/bigbabol1/AI-plugin/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/bigbabol1/AI-plugin/test.yml?branch=main&style=flat-square&label=tests" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue?style=flat-square" alt="License: PolyForm Noncommercial 1.0.0"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#choosing-a-model">Choosing a model</a> ·
  <a href="#features-in-depth">Features</a> ·
  <a href="#configuration-reference">Configuration</a> ·
  <a href="#troubleshooting">Troubleshooting</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

---

AI Plugin makes any OpenAI-compatible model the **Assist conversation agent** for your home. That includes Ollama, llama.cpp, LM Studio, OpenAI and xAI. It handles device control, sensor questions, timers, web search and memory, by voice or in text.

It is designed around a constraint most agents ignore: **a 7–12B model on an 8 GB GPU**. Common commands skip the LLM entirely. The prompt is kept stable so Ollama can reuse its cache. Replies start speaking after the first sentence. When a model claims it did something it never did, the plugin catches it.

## Highlights

**🎙️ Built for voice**
- Streaming replies: voice pipelines with streaming TTS start talking after the first sentence.
- Follow-up listening: always, never, or **let the model decide** by asking a question.
- Self-echo filtering for satellites whose replies play on a separate speaker.
- First-class voice timers, with an optional announce mode for any speaker.

**⚡ Fast on local hardware**
- A deterministic shortcut layer answers common commands in **~0.01–0.05 s** without calling the LLM.
- A cache-stable prompt layout reuses Ollama's prefix cache on every turn.
- On-demand entity discovery tools instead of a YAML entity dump in the prompt.
- Summarization runs in the background, never on the turn that hits the limit.

**🛡️ Grounded, not imagined**
- State questions, actions and time-sensitive questions are checked against the tools the model actually called. A reply made up from nothing gets one corrective re-run.
- Promised actions without a tool call are replaced with an honest failure notice.
- The config flow rejects models that cannot call tools, and warns when the context window is larger than the model supports.

**🧩 Extensible**
- Web search via DuckDuckGo (zero config), Brave, Tavily or SearXNG, plus `browse_url` for reading one page.
- MCP servers over HTTP and stdio, including Home Assistant's own MCP server.
- An `ai_task` entity so automations can use the same model.
- Per-user long-term memory (`remember` / `recall` / `forget`).
- **7 languages**: English, German, French, Spanish, Portuguese, Polish and Dutch. A new language is a single YAML file.

> [!NOTE]
> **Tested end-to-end with Ollama.** The OpenAI-compatible endpoint should work with the other backends, but only Ollama has been run against the prompt suites below. Reports from OpenAI, xAI, LM Studio or llama.cpp setups are welcome.

## Quick start

### Requirements

- Home Assistant **2025.7.0** or newer
- [HACS](https://hacs.xyz/docs/use/)
- A reachable LLM endpoint:

| Backend | Provider URL | API key |
|---------|--------------|---------|
| [Ollama](https://ollama.com) *(recommended)* | `http://<host>:11434/v1` | — |
| LM Studio | `http://<host>:1234/v1` | — |
| llama.cpp server | `http://<host>:8080/v1` | — |
| OpenAI | `https://api.openai.com/v1` | required |
| xAI Grok | `https://api.x.ai/v1` | required |

### Install

[![Open your Home Assistant instance and open this repository in HACS.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=bigbabol1&repository=AI-plugin&category=integration)

1. In **HACS → ⋮ → Custom repositories**, add `https://github.com/bigbabol1/AI-plugin` with category **Integration**. You can also use the button above.
2. Install **AI Plugin**, then **restart Home Assistant**.
3. Add the integration:

   [![Open your Home Assistant instance and start setting up AI Plugin.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ai_plugin)

4. Go to **Settings → Voice assistants**, open your assistant and set **Conversation agent** to *AI Plugin*.

Every setting can be changed later under **Settings → Devices & services → AI Plugin → Configure**.

## Choosing a model

**Tool-calling quality matters more than any benchmark score.** A model that hedges or asks clarifying questions instead of acting is a poor fit for a home agent, however well it scores elsewhere.

### Recommended for 8 GB of VRAM

| Model | Why | Median turn |
|-------|-----|-------------|
| **`hf.co/unsloth/gemma-4-12B-it-GGUF:IQ4_XS`** | Same Gemma 4 12B at 4 bits. Holds up in long conversations where the 3-bit quant stalls, at the same generation speed. | — |
| **`hf.co/unsloth/gemma-4-12B-it-GGUF:UD-Q3_K_XL`** | The previous recommendation: 0.4 GB smaller and fully in VRAM, but in long voice conversations it can stall on a turn and answer nothing. | ~3.6 s |
| **`qwen3.5:9b`** | Faster, says so when it lacks data, validated against the voice suite | ~2.9 s |

```bash
ollama pull hf.co/unsloth/gemma-4-12B-it-GGUF:IQ4_XS
```

IQ4_XS has not yet been run through the voice suite, so it has no median turn time yet. In the stall tests below it answered cached turns in about a second.

Avoid `granite4.1:8b` and `lfm2.5:8b` on non-English installs. Check `ollama ps` at your real context window. If part of the model runs on the CPU, measure tokens per second before keeping it. On Ollama 0.40 IQ4_XS shows 5 % CPU at a 16K context with no measurable slowdown, but a larger share can multiply latency.

<details>
<summary><b>Benchmark details</b> (RTX 3060 Ti 8 GB, Ollama)</summary>

<br>

Pass counts alone hide *how* a model fails, so these notes separate honest misses from confident fabrications. In a home agent, that difference matters most.

- **Voice suite (July 2026, plugin v0.9.35).** 29 read-only prompts through HA Assist in voice mode, covering clock and sun, room sensors, "which lights are on", inventory, climate, local and named-place weather, web search, bare-noun sanity checks and media.
  - `qwen3.5:9b` scored 93 %, and every miss was an honest one.
  - `qwen3:8b` and `qwen3-abliterated:8b` reached 86–90 % but fabricated answers.
  - `mistral:7b` scored 86 % and `gemma4:e2b` 69 %.
  - Cloud `qwen3.5:397b` (100 %) served as the ceiling.
- **Adversarial tool calling (August 2026).** 10 German edge cases × 3 runs.
  - `gemma-4-12B UD-Q3_K_XL` scored 100 % with exact area names; `qwen3.5:9b` scored 70 %.
  - `granite4.1:8b` and `lfm2.5:8b` corrupted or translated non-English room names.
  - Larger Gemma 4 12B quants (IQ4_XS, Q4_K_M) spilled onto the CPU at a 16K context on the Ollama of the time. See October for IQ4_XS on Ollama 0.40.
- **Multi-turn conversations (August 2026).** 11 German turns × 2 passes on `gemma-4-12B UD-Q3_K_XL`, with every action verified in HA's state history.
  - Ambiguous names, ellipsis, pronouns, two devices in one turn, self-correction and area references all passed. Median turn: 4.5 s.
  - Changing `top_p` or temperature (0.3–1.0), using a 32K window, or enabling schema pruning changed nothing. Raising Max Tokens to 1024 only made failing turns slower.
- **LFM2.5 QAD checkpoints (August 2026).** Very fast, but they did not converge in the plugin's multi-step tool loop.
- **Stalled turns in long conversations (October 2026, Ollama 0.40.4).** Repeated voice requests like "too bright" and "a bit dark in here" in one long conversation sometimes made `UD-Q3_K_XL` produce nothing. With thinking off, it slipped into hidden thinking that never ended. It used all of Max Tokens, returned no text and no tool call, and after ~15 s the user heard the fallback. Five requests captured from real sessions, each in two prompt layouts, plus one reconstructed turn, were replayed 3 times each (11 cases):
  - `UD-Q3_K_XL`: 9 of 33 runs empty.
  - `IQ4_XS`: 0 of 33 empty, all with the right tool (dim, brighten, or a Wikipedia lookup for a trivia question).
  - Generation speed was the same for both: 39 tokens/s on an RTX 3060 Ti.
  - Neither a larger Max Tokens (2048) nor Ollama's built-in Gemma 4 template fixed `UD-Q3_K_XL`. That points to the 3-bit quant: Google's 4-bit QAT build of the same model did not stall either.

</details>

### Recommended settings

| Setting | Value |
|---------|-------|
| Model | `hf.co/unsloth/gemma-4-12B-it-GGUF:IQ4_XS`, or `UD-Q3_K_XL` / `qwen3.5:9b` if it does not fit |
| Temperature | `1.0` for gemma-4-12B (0.3–1.0 made no difference to the actions taken, and 0.3 did not prevent the stalls). `qwen3.5:9b` was benchmarked at 0.2. |
| top_p | `0.4` |
| Context Window | **`16384`** minimum. Stay under ~24 000 on 8 GB with the default KV cache; 32K fits with `OLLAMA_KV_CACHE_TYPE=q8_0`. |
| Max Tokens (`num_predict`) | `512`. Raising it makes failing turns slower, not better. |
| Keep-alive | `30m`, or `-1` on a dedicated GPU |
| Web Search | DuckDuckGo (default), or Tavily/Brave with a key |

## Features in depth

### Follow-up listening

After a spoken reply, the satellite can reopen its microphone so you can answer without the wake word. **Listen for a follow-up after voice replies** has three modes:

| Mode | Behaviour |
|------|-----------|
| **Always** *(default)* | The microphone reopens after every reply. |
| **Let the AI decide** | The microphone reopens only when the reply **ends with a question**. "Which room do you mean?" keeps listening. "The kitchen light is on." ends the turn. |
| **Never** | Every turn ends after the reply. |

In *Let the AI decide*, the question mark *is* the model's decision. This is the same signal Home Assistant's built-in LLM agents use. The voice prompt gains one rule: ask only when an answer is needed (a missing detail, a choice, a confirmation), and never close with filler like "Anything else?". So the setting adds no extra tool call or round trip, and it works with streaming.

In every mode, the conversation still ends on a close phrase (*"thanks"*, *"bye"*, *"das war's"*, …), on a detected self-echo, and after a run of rapid chained turns.

<details>
<summary><b>Satellites that play replies on a separate speaker</b></summary>

<br>

A satellite that routes its reply audio to an external media player (Google/Cast, *Mic to MediaPlayer*) can hear its own TTS and treat it as a new turn. That is an acoustic feedback loop: the assistant talks to itself. Three tools address it:

- **Self-echo filter** (Advanced → *Ignore the satellite hearing its own reply*, on by default). The plugin remembers what it just said on each device and silently drops any incoming voice turn that replays it: 70 % of the turn's word pairs, in order, within 30 s. A recognizable command or a bare "yes" is never dropped. The session stays open, so real follow-ups still work. This is the recommended fix.
- **Quiet gap before listening again.** Above 0 s, the plugin reopens the microphone itself once the reply has finished playing in the room, instead of when the satellite's own playback ends.
- **Force-end conversations on these satellites** (device list). A blunter guard that ends the session after one turn on the listed devices. Use it if STT garbles the echo too much for the filter to catch it.

</details>

### Voice timers

Timers are first-class tools: `start_timer`, `cancel_timer`, `pause_timer`, `unpause_timer`, `increase_timer`, `decrease_timer` and `timer_status`. A deterministic safety net treats the spoken utterance as the ground truth for the duration, so "10 seconds" never becomes 10 minutes, even if the model fills a slot wrongly.

- **Default.** HA sends the timer to the satellite, which shows the countdown and plays its ring locally. This needs timer-capable firmware, which HA Voice PE and current ESPHome satellites have.
- **Announce mode** (Advanced → *Announce timers instead of on-device ring*). HA's TimerManager calls the plugin back when the timer expires. The plugin then plays a localized announcement ("Timer pasta is up.") through `mic_to_mediaplayer.announce` if that integration is installed, and `assist_satellite.announce` otherwise. It works on **any** device, including satellites whose own speaker is silenced or that have no timer firmware.
  - Trade-off: no countdown LEDs and no local ring.
  - Like all HA voice timers, these do not survive an HA restart.

### Long-term memory

The assistant remembers facts across conversations and across restarts:

- *"Remember that I take my coffee black."* → `remember`
- *"What did I tell you about my coffee?"* → answered from memory
- *"Forget that I take my coffee black."* → `forget`

**Recall is usually free.** Saved facts are loaded into a `[USER FACTS]` block at the start of every turn. That block is a late, cache-stable system message, so the model answers "what's my…" questions without a tool round trip, and each round trip is a full extra generation pass on a small model.

**Per-user and plain text.** Facts are stored at `<config>/.ai_plugin_memory_<user_id>.json`, one file per HA user (`…_anonymous.json` for unauthenticated turns). The files are readable, so don't ask it to remember secrets.

**`forget` won't delete the wrong fact.** It removes by the fact's number first and falls back to a fuzzy match. If the fact isn't stored, it says so instead of guessing.

### AI Tasks for automations

The integration provides an `ai_task` entity, so automations and scripts can use the same model:

```yaml
action: ai_task.generate_data
data:
  task_name: morning summary
  instructions: Summarize today's calendar in two sentences.
  entity_id: ai_task.ai_plugin
```

With a `structure` schema, the model returns matching JSON. Code fences are tolerated, and invalid JSON raises a clear error instead of returning garbage.

### Web search

| Backend | Cost | API key | Notes |
|---------|------|---------|-------|
| **DuckDuckGo** *(default)* | Free | none | Zero config. Can be rate-limited. Ambiguous place names are disambiguated with your city and country. |
| Brave Search | Free tier / paid | [api.search.brave.com](https://api.search.brave.com/app/) | Reliable |
| Tavily | Free tier / paid | [tavily.com](https://tavily.com/) | Best quality, AI-optimized |
| SearXNG | Free (self-hosted) | instance URL | Full control, no third-party traffic |

To read one specific page, the model uses `browse_url`. It fetches through **Jina Reader** (`r.jina.ai`), so that URL is sent to a third party. Apart from your chosen search backend, this is the only outbound connection the integration makes.

### MCP servers

You can add any MCP tool server over HTTP or stdio. Presets are included for HA's built-in MCP server (recommended: it provides `HassTurnOn` / `HassLightSet`-style intents), time, fetch, SQLite and more. MCP calls are capped at 15 s, so a hung server cannot stall a voice reply.

## How it stays fast

On a 7–12B model, latency is won or lost in three places:

1. **Skip the LLM when possible.** The shortcut layer handles clock time, sunrise and sunset, "temperature in the bedroom", single-device on/off, covers, media transport, volume and mute. It calls HA registries and services directly, in all seven languages. Misses and anything ambiguous fall through to the LLM; the shortcut never guesses.
2. **Reuse Ollama's prompt prefix cache.** The system prompt and tool list stay byte-stable across turns. Volatile context (`[CURRENT TIME]`, `[USER FACTS]`, `[LAST ACTION]`) goes in a late message, so each request pre-fills only the new tail, not the whole 3–6K-token prompt.
3. **Speak before generation finishes.** Replies stream sentence by sentence into HA's chat log. Each sentence passes the same cleanup as the final reply, and the last sentence is held back. Streaming switches off on any turn where a verifier might still rewrite the reply.

<details>
<summary><b>Ollama tuning notes</b></summary>

<br>

- **Context window: one number.** The plugin uses Ollama's native `/api/chat` and sends `num_ctx` on every request. The *Context Window* setting therefore sets both the plugin's token budget and the context Ollama loads the model with. No Modelfile changes are needed.
- **Temperature.** If a model asks clarifying questions instead of acting, lower it to 0.1–0.3.
- **Reasoning models** (qwen3, deepseek-r1). The plugin sends `think: false` so answers land in `content`. Requires Ollama 0.20+.
- **Keep-alive.** By default Ollama unloads a model after 5 minutes idle. On 8 GB, reloading a 6.5 GB model takes **~16 s**. Set *Ollama keep-alive* (`30m`, `24h`, or `-1`); it is sent per request and needs no server configuration.

</details>

## Configuration reference

| Option | Default | What it does |
|--------|---------|--------------|
| System prompt | persona template | Appended to the built-in prompt (never replaces it) |
| Context Window | `16384` | Token budget **and** Ollama `num_ctx` |
| Summarization | on | Condenses old turns in the background near the limit |
| Voice mode | off | Forces the compact voice prompt, even for text |
| Listen for a follow-up | Always | Always / Let the AI decide / Never (see [Follow-up listening](#follow-up-listening)) |
| Quiet gap before listening again | `0` s | Above 0, the plugin reopens the microphone itself once the room is quiet |
| Enable thinking | off | Lets reasoning models emit chain-of-thought (slower; disables streaming) |
| Max tool iterations | `10` | Cap on LLM↔tool round trips per turn |
| Timeout | `30` s | Per LLM call. MCP tools have their own 15 s cap. |
| Temperature / top_p / Max tokens | unset | Sampling. 1.0 / 0.4 / 512 recommended for gemma-4-12B. |
| Ollama keep-alive | unset | Keeps the model in VRAM between requests |
| Prune tool schemas | off | Per-message schema pruning. Defeats the prefix cache, so leave it off. |
| Announce timers | off | Timer completions play on a media player instead of the on-device ring |
| Ignore the satellite hearing its own reply | on | Self-echo filter |
| Force-end conversations on these satellites | none | Feedback-loop guard for setups with TTS on a separate speaker |
| Location bias | on | Scopes "near me" searches to your home (fails open) |
| Location entity | unset | Live location source instead of HA's home coordinates |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Cannot connect" during setup | The Provider URL must end in `/v1` and be reachable from the HA host. |
| "Model reports no tool-calling capability" | The config flow checked Ollama's `/api/show`. Pick a tool-capable model. |
| Model asks questions instead of acting | Lower the temperature to 0.1–0.3, or use a stronger tool-calling model. |
| Slow first reply after a quiet period | Set **Ollama keep-alive**. |
| Replies don't start speaking early | Streaming needs the Ollama backend and a streaming-capable TTS engine. Other setups get the full reply at once. |
| *Let the AI decide* listens after every reply | The model is closing with filler questions. Add "Do not end with a question unless you need an answer" to the system prompt, or try a stronger model. |
| The assistant answers itself in a loop | Replies are playing on a separate speaker. See [Follow-up listening](#follow-up-listening). |
| Timer LEDs count down but there's no sound | The ring plays on the satellite's own speaker. Enable **Announce timers**. |
| Empty replies from reasoning models | Upgrade Ollama to 0.20+. |

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements_test.txt
.venv/bin/python -m pytest
```

The unit suite runs against stubbed Home Assistant modules, so no HA install is needed. The suite mocks every LLM call. **`tests/eval/`** measures the real thing: it drives a live install through `/api/conversation/process`, scores replies against per-prompt rubrics, checks actions against live entity state, and reports pass rates and latency percentiles. See [`tests/eval/README.md`](tests/eval/README.md).

**Adding a language** means one YAML file: copy `i18n/en.yaml`, translate it, and open a PR. See [`i18n/CONTRIBUTING.md`](custom_components/ai_plugin/i18n/CONTRIBUTING.md).

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Free to use, change and share for any
noncommercial purpose: your own home, hobby projects, study, research, and charities, schools or
public bodies. Commercial use, including selling it or bundling it into a paid product or service,
needs a separate license; open an issue to ask. Contributions are accepted under the same terms.

Release history is in [CHANGELOG.md](CHANGELOG.md).
