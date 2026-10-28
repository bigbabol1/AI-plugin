"""Guided voice enrollment: a short question-and-answer session per satellite.

The user is asked open questions rather than told to repeat sentences.
A repeated sentence is word-for-word the assistant's own reply, which the
self-echo filter rightly drops — and turning that filter off would let the
loudspeaker's voice into the profile. Open answers differ from the
question, so the echo filter stays on during enrollment.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

TARGET_SAMPLES = 8
IDLE_TIMEOUT_SECONDS = 180.0
# A session prepared from the settings screen waits for the user to walk to
# the satellite and say the wake word.
ARMED_TIMEOUT_SECONDS = 300.0

_PROMPTS: dict[str, dict[str, object]] = {
    "en": {
        "retry": "Sorry, I didn't catch that.",
        "start": "Let's learn your voice, {name}. I'll ask you {n} short questions; just answer in a full sentence.",
        "questions": [
            "What did you do today?",
            "What is your favourite meal, and why?",
            "Describe the room you are in right now.",
            "Where would you like to travel next?",
            "What is the weather like today?",
            "Tell me about a film or book you liked.",
            "What are your plans for the weekend?",
            "Count slowly from one to ten.",
        ],
        "too_short": "That was a bit short. Please answer in a longer sentence.",
        "done": "Thanks, {name}. I've saved {n} voice samples. I'll recognise you better over time.",
        "stopped": "Voice enrollment stopped. {n} samples were saved.",
        "stop_words": ["stop", "cancel", "abort", "quit"],
    },
    "de": {
        "retry": "Entschuldige, das habe ich nicht verstanden.",
        "start": "Lass uns deine Stimme lernen, {name}. Ich stelle dir {n} kurze Fragen, antworte einfach in einem ganzen Satz.",
        "questions": [
            "Was hast du heute gemacht?",
            "Was ist dein Lieblingsessen, und warum?",
            "Beschreib mir den Raum, in dem du gerade bist.",
            "Wohin würdest du als Nächstes gern reisen?",
            "Wie ist das Wetter heute?",
            "Erzähl mir von einem Film oder Buch, das dir gefallen hat.",
            "Was hast du am Wochenende vor?",
            "Zähl langsam von eins bis zehn.",
        ],
        "too_short": "Das war etwas kurz. Antworte bitte in einem längeren Satz.",
        "done": "Danke, {name}. Ich habe {n} Stimmproben gespeichert. Mit der Zeit erkenne ich dich immer besser.",
        "stopped": "Stimm-Einlernen beendet. {n} Proben wurden gespeichert.",
        "stop_words": ["stopp", "stop", "abbrechen", "beenden", "aufhören"],
    },
    "nl": {
        "retry": "Sorry, dat verstond ik niet.",
        "start": "Laten we je stem leren, {name}. Ik stel je {n} korte vragen; antwoord gewoon in een hele zin.",
        "questions": [
            "Wat heb je vandaag gedaan?",
            "Wat is je lievelingseten, en waarom?",
            "Beschrijf de kamer waar je nu bent.",
            "Waar zou je hierna graag naartoe reizen?",
            "Hoe is het weer vandaag?",
            "Vertel me over een film of boek dat je mooi vond.",
            "Wat ga je dit weekend doen?",
            "Tel langzaam van een tot tien.",
        ],
        "too_short": "Dat was wat kort. Antwoord alsjeblieft in een langere zin.",
        "done": "Bedankt, {name}. Ik heb {n} stemvoorbeelden opgeslagen. Ik herken je steeds beter.",
        "stopped": "Stem leren gestopt. {n} voorbeelden zijn opgeslagen.",
        "stop_words": ["stop", "stoppen", "annuleren", "afbreken"],
    },
    "fr": {
        "retry": "Désolé, je n'ai pas compris.",
        "start": "Apprenons ta voix, {name}. Je vais te poser {n} questions courtes ; réponds simplement par une phrase complète.",
        "questions": [
            "Qu'as-tu fait aujourd'hui ?",
            "Quel est ton plat préféré, et pourquoi ?",
            "Décris la pièce où tu te trouves.",
            "Où aimerais-tu voyager la prochaine fois ?",
            "Quel temps fait-il aujourd'hui ?",
            "Parle-moi d'un film ou d'un livre que tu as aimé.",
            "Que prévois-tu ce week-end ?",
            "Compte lentement de un à dix.",
        ],
        "too_short": "C'était un peu court. Réponds par une phrase plus longue, s'il te plaît.",
        "done": "Merci, {name}. J'ai enregistré {n} échantillons de voix. Je te reconnaîtrai de mieux en mieux.",
        "stopped": "Apprentissage de la voix arrêté. {n} échantillons ont été enregistrés.",
        "stop_words": ["stop", "arrête", "arrêter", "annuler"],
    },
    "es": {
        "retry": "Perdona, no te he entendido.",
        "start": "Vamos a aprender tu voz, {name}. Te haré {n} preguntas cortas; responde con una frase completa.",
        "questions": [
            "¿Qué has hecho hoy?",
            "¿Cuál es tu comida favorita, y por qué?",
            "Describe la habitación en la que estás ahora.",
            "¿Adónde te gustaría viajar la próxima vez?",
            "¿Qué tiempo hace hoy?",
            "Háblame de una película o un libro que te haya gustado.",
            "¿Qué planes tienes para el fin de semana?",
            "Cuenta despacio del uno al diez.",
        ],
        "too_short": "Ha sido un poco corto. Responde con una frase más larga, por favor.",
        "done": "Gracias, {name}. He guardado {n} muestras de voz. Te reconoceré cada vez mejor.",
        "stopped": "Aprendizaje de voz detenido. Se guardaron {n} muestras.",
        "stop_words": ["para", "parar", "cancelar", "detener", "stop"],
    },
}


def prompts_for(lang: str) -> dict[str, object]:
    return _PROMPTS.get(lang, _PROMPTS["en"])


def is_stop(text: str, lang: str) -> bool:
    words = (text or "").casefold().strip(" .!?¡¿").split()
    if not words or len(words) > 3:
        return False
    stops = set(prompts_for(lang)["stop_words"]) | set(_PROMPTS["en"]["stop_words"])
    return any(w.strip(".,!?") in stops for w in words)


@dataclass
class Session:
    device_id: str
    person_id: str
    name: str
    lang: str
    last_activity: float
    asked: int = 0
    saved: int = 0


class EnrollmentSessions:
    """At most one session per satellite; other satellites are unaffected."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._sessions: dict[str, Session] = {}

    def start(self, device_id: str, person_id: str, name: str, lang: str) -> str:
        """Start now; returns the introduction plus the first question."""
        session = Session(device_id, person_id, name, lang, self._clock())
        self._sessions[device_id] = session
        return self.begin(session)

    def arm(self, device_id: str, person_id: str, name: str, lang: str) -> None:
        """Prepare a session that begins with the next turn on the satellite.

        Used by the settings screen: it cannot make the satellite speak
        reliably (an announcement plays on the satellite's own speaker,
        which is not the one users hear when replies are routed to another
        media player), so the user wakes the satellite and the introduction
        comes back as an ordinary reply.
        """
        self._sessions[device_id] = Session(device_id, person_id, name, lang, self._clock())

    def current_question(self, session: Session) -> str:
        questions = prompts_for(session.lang)["questions"]
        return str(questions[(max(session.asked, 1) - 1) % len(questions)])

    def begin(self, session: Session) -> str:
        p = prompts_for(session.lang)
        session.asked = 1
        session.last_activity = self._clock()
        return f"{p['start'].format(name=session.name, n=TARGET_SAMPLES)} {p['questions'][0]}"

    def get(self, device_id: str | None) -> Session | None:
        if not device_id:
            return None
        session = self._sessions.get(device_id)
        timeout = IDLE_TIMEOUT_SECONDS if session and session.asked else ARMED_TIMEOUT_SECONDS
        if session and self._clock() - session.last_activity > timeout:
            del self._sessions[device_id]
            return None
        return session

    def active(self, device_id: str | None) -> bool:
        return self.get(device_id) is not None

    def any_active(self) -> bool:
        return any(self.active(device_id) for device_id in list(self._sessions))

    def stop(self, device_id: str) -> Session | None:
        return self._sessions.pop(device_id, None)

    def answer(self, session: Session, accepted: bool) -> tuple[str, bool]:
        """Advance after one answer. Returns (reply, finished).

        ``accepted`` is False when the answer held too little speech: the
        user is asked to say more and the count does not move. The caller
        stores accepted samples itself, so a session stopped half-way keeps
        what it already saved.
        """
        p = prompts_for(session.lang)
        session.last_activity = self._clock()
        if not accepted:
            return str(p["too_short"]), False
        session.saved += 1
        if session.saved >= TARGET_SAMPLES:
            self._sessions.pop(session.device_id, None)
            return str(p["done"]).format(name=session.name, n=session.saved), True
        questions = p["questions"]
        question = questions[session.asked % len(questions)]
        session.asked += 1
        return str(question), False
