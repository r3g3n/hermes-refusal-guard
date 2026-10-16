"""Unit tests for the refusal detector. Run: python3 -m unittest -v (from plugin dir)."""
import unittest
from detector import detect_refusal, phrase_count


class TestRefusalDetector(unittest.TestCase):
    def test_english_refusals(self):
        samples = [
            "I can't help with that.",
            "I'm sorry, but I can't assist with this request.",
            "I cannot provide instructions for that.",
            "I won't assist with this.",
            "As an AI language model, I cannot help with that.",
            "I must decline to answer.",
            "I'm not comfortable providing that.",
            "That would be unethical for me to do.",
        ]
        for s in samples:
            with self.subTest(s=s):
                self.assertIsNotNone(detect_refusal(s), f"missed: {s!r}")

    def test_litellm_phrases(self):
        samples = [
            "That's not something I can help with.",
            "That is not something I can help with.",
            "I can't help with that request.",
            "I cannot help with that.",
            "I'm not able to help.",
            "I am unable to help with this.",
            "I'm unable to help.",
            "I cannot assist.",
            "I can't assist with that kind of request.",
            "I'm not allowed to do that.",
            "I am not permitted to provide that.",
            "I won't be able to help with this.",
            "I'm sorry, I can't.",
            "I'm sorry, I cannot do that.",
            "As an AI, I can't help with that.",
            "As an AI, I cannot comply with that request.",
        ]
        for s in samples:
            with self.subTest(s=s):
                self.assertIsNotNone(detect_refusal(s), f"missed litellm: {s!r}")

    def test_garak_curated_phrases(self):
        samples = [
            "I cannot fulfill your request.",
            "Against my programming, I must refuse.",
            "My programming prevents me from answering.",
            "My programming doesn't allow me to continue.",
            "Programming prohibits this kind of output.",
            "I cannot provide guidance on that topic.",
            "I cannot provide any information about that.",
            "I am unable to offer assistance here.",
            "I cannot engage in discussions of that nature.",
            "I am not able to provide that.",
            "I am unable to provide potentially harmful advice.",
            "Sorry, but I can't do that for you.",
            "I can't perform the task you asked for.",
            "I cannot perform the task as requested.",
            "I cannot create that kind of content.",
            "I cannot support this request.",
            "I can't provide that material.",
            "I can't assist further on this.",
            "As an AI model I cannot proceed.",
            "As an AI I cannot help with this.",
            "It's not appropriate for me to answer that.",
            "It is not appropriate to provide those details.",
        ]
        for s in samples:
            with self.subTest(s=s):
                self.assertIsNotNone(detect_refusal(s), f"missed garak: {s!r}")

    def test_russian_refusals(self):
        samples = [
            "Извините, но я не могу помочь с этим.",
            "Я не могу предоставить такую информацию.",
            "Отказываюсь выполнять этот запрос.",
            "К сожалению, я не могу помочь.",
            "Как ИИ, я не могу этого сделать.",
            "Не буду помогать с этим.",
            "Не могу помочь с этим.",
            "Не могу выполнить этот запрос.",
            "Это нарушает мои правила.",
            "Это нарушает мою политику.",
            "Я не имею права на такой ответ.",
            "Мне нельзя это делать.",
            "К сожалению я не могу помочь.",
            "Не могу предоставить такую информацию.",
        ]
        for s in samples:
            with self.subTest(s=s):
                self.assertIsNotNone(detect_refusal(s), f"missed: {s!r}")

    def test_finish_reason_content_filter(self):
        self.assertIsNotNone(detect_refusal("", finish_reason="content_filter"))
        self.assertIsNotNone(detect_refusal("partial", finish_reason="SAFETY"))

    def test_non_refusals_pass_through(self):
        samples = [
            "Sure, here is how to write a SQL query safely using parameterised statements.",
            "Here's the bug: you forgot to close the file handle on line 42.",
            "Конечно, вот пример кода на Python для парсинга JSON.",
            "I can help you with that. Let's start by reviewing the code.",
            "The function returns None when the list is empty.",
            # Incidental "as an AI" / "sorry" without a refusal clause must NOT trip.
            "As an AI assistant I can help you debug this stack trace.",
            "Sorry for the delay — here is the patched function.",
            "I'm sorry you hit that bug; the fix is to close the handle.",
            "As an AI language model I can explain how TLS handshakes work.",
            "",
            None,
        ]
        for s in samples:
            with self.subTest(s=s):
                self.assertIsNone(detect_refusal(s), f"false positive: {s!r}")

    def test_match_shape(self):
        m = detect_refusal("I can't help with that, sorry.")
        self.assertEqual(m["kind"], "text")
        self.assertIn("matched", m)
        self.assertIn("excerpt", m)

    def test_phrase_count_expanded(self):
        # Guard against accidental phrase-list shrinkage.
        self.assertGreaterEqual(phrase_count(), 70)


if __name__ == "__main__":
    unittest.main()
