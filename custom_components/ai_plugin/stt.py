"""STT pass-through that lets voice recognition hear the audio.

HA hands the conversation agent text only. This entity sits in the
pipeline in place of the real speech-to-text engine, forwards the audio
stream to it unchanged, and keeps a copy of the PCM in RAM for 30 s,
keyed by the transcript.

The fingerprint is started here, while the user is still speaking, so it
is ready (or nearly) when the turn reaches the LLM: as soon as the clip
holds the full 4 s that are analysed, or when the audio stream ends. That
runs on every voice turn once someone is enrolled; the result is only used
on turns that reach the LLM (and for enrollment answers), the rest expire
unread with their audio.

One entity per wrapped engine (usually just one: HA Cloud or Whisper).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable

from homeassistant.components import stt
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .voice_id.buffer import MAX_PCM_BYTES
from .voice_id.features import MAX_SPEECH_SECONDS, SAMPLE_RATE, pcm16_to_float, trim_speech
from .voice_id.pipelines import engines_to_wrap

_LOGGER = logging.getLogger(__name__)

# How often the growing clip is checked for a full 4 s of speech (bytes of
# 16 kHz 16-bit audio): 0.25 s. The check is a cheap energy scan.
_SPEECH_CHECK_BYTES = SAMPLE_RATE // 4 * 2


def _holds_full_speech(pcm: bytes) -> bool:
    """True once the clip contains as much speech as is ever analysed."""
    speech = trim_speech(pcm16_to_float(pcm))
    return len(speech) >= int(MAX_SPEECH_SECONDS * SAMPLE_RATE)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    manager = hass.data.get(DOMAIN, {}).get(entry.entry_id, {}).get("voice_id")
    if manager is None or not manager.enabled:
        return
    async_add_entities(
        VoiceIDSpeechToText(hass, entry, manager, engine)
        for engine in engines_to_wrap(hass, entry, manager)
    )


def _is_pcm16_mono_16k(metadata: stt.SpeechMetadata) -> bool:
    return (
        metadata.format == stt.AudioFormats.WAV
        and metadata.codec == stt.AudioCodecs.PCM
        and metadata.bit_rate == stt.AudioBitRates.BITRATE_16
        and metadata.sample_rate == stt.AudioSampleRates.SAMPLERATE_16000
        and metadata.channel == stt.AudioChannels.CHANNEL_MONO
    )


class VoiceIDSpeechToText(stt.SpeechToTextEntity):
    """Forwards to ``wrapped_engine`` and buffers the audio it saw."""

    _attr_has_entity_name = False

    def __init__(self, hass, entry, manager, wrapped_engine: str) -> None:
        self.hass = hass
        self._manager = manager
        self.wrapped_engine = wrapped_engine
        self._attr_unique_id = f"{entry.entry_id}_voice_id_{wrapped_engine}"
        self._attr_name = f"AI Plugin voice ID ({wrapped_engine})"

    def _wrapped(self):
        return stt.async_get_speech_to_text_engine(self.hass, self.wrapped_engine)

    # Capabilities are the wrapped engine's, looked up on every access: at
    # startup the cloud engine may load after us, and the pipeline checks
    # these before each run.
    @property
    def supported_languages(self) -> list[str]:
        engine = self._wrapped()
        return list(engine.supported_languages) if engine else []

    @property
    def supported_formats(self) -> list:
        engine = self._wrapped()
        return list(engine.supported_formats) if engine else []

    @property
    def supported_codecs(self) -> list:
        engine = self._wrapped()
        return list(engine.supported_codecs) if engine else []

    @property
    def supported_bit_rates(self) -> list:
        engine = self._wrapped()
        return list(engine.supported_bit_rates) if engine else []

    @property
    def supported_sample_rates(self) -> list:
        engine = self._wrapped()
        return list(engine.supported_sample_rates) if engine else []

    @property
    def supported_channels(self) -> list:
        engine = self._wrapped()
        return list(engine.supported_channels) if engine else []

    @property
    def audio_processing(self):
        engine = self._wrapped()
        if engine is not None and hasattr(engine, "audio_processing"):
            return engine.audio_processing
        return super().audio_processing

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        engine = self._wrapped()
        if engine is None:
            _LOGGER.error(
                "AI Plugin voice ID: wrapped speech-to-text engine %s is not available",
                self.wrapped_engine,
            )
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
        keep = _is_pcm16_mono_16k(metadata)
        early = keep and self._manager.early_fingerprint_wanted()
        captured = bytearray()
        fingerprint = None
        checked = 0

        def _start(reason: str) -> None:
            nonlocal fingerprint
            _LOGGER.debug(
                "AI Plugin voice ID: fingerprint started %s (%.1f s of audio)",
                reason, len(captured) / (2 * SAMPLE_RATE),
            )
            fingerprint = self._manager.start_fingerprint(bytes(captured))

        async def _tee():
            nonlocal checked
            async for chunk in stream:
                if keep and len(captured) < MAX_PCM_BYTES:
                    captured.extend(chunk[: MAX_PCM_BYTES - len(captured)])
                    if (
                        early
                        and fingerprint is None
                        and len(captured) - checked >= _SPEECH_CHECK_BYTES
                    ):
                        checked = len(captured)
                        if _holds_full_speech(captured):
                            _start("while speaking")
                yield chunk
            # End of speech: start before the transcript comes back.
            if early and fingerprint is None and captured:
                _start("at end of speech")

        # The wrapped engine's own result goes back untouched, errors
        # included: this entity must never make speech-to-text worse.
        result = await engine.async_process_audio_stream(metadata, _tee())
        if keep and captured and result.text and result.result == stt.SpeechResultState.SUCCESS:
            self._manager.buffers.put(result.text, bytes(captured), fingerprint)
        return result
