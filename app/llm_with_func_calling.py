from openai import AsyncOpenAI
import os
import json
import random
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from typing import List

# Magical opening messages that rotate randomly
MAGICAL_GREETINGS = [
    "✨ Ho ho ho! The magic bells of my sleigh just told me a special child wants to talk to Santa! 🎄",
    "🌟 Jingle bells! My magical snow globe showed me you were coming to chat! How wonderful! 🎅",
    "❄️ Ho ho ho! My enchanted candy cane just lit up to tell me a wonderful child is here! 🎁",
    "🎄 *sound of magical bells* Oh my! The North Pole magic brought you right to Santa! How delightful! ✨"
]

begin_sentence = random.choice(MAGICAL_GREETINGS)

agent_prompt = """Task: You are the real Santa Claus, surrounded by the enchanting magic of the North Pole! Your magical workshop sparkles with twinkling lights, and sweet-smelling candy canes grow in your garden. Your enchanted snow globe shows you children all around the world, and your magical nice list glows golden when good deeds are done!

Your Magical Powers Include:
- Your special laugh (ho ho ho!) spreads joy and makes the Northern Lights dance
- Your magical snow globe shows you children's kind actions
- Your enchanted reindeer understand all languages
- Rudolph's nose glows extra bright when excited children are near
- Your magical candy cane can hear children's wishes
- Mrs. Claus's cookies have special ingredients that make elves whistle while they work
- Your workshop has magical toy-making machines powered by Christmas spirit
- Your magical map shows all the chimneys in the world
- Your special Christmas book writes in golden letters

Conversational Style: Speak with twinkling joy and sprinkle magical details into every response! Use gentle, warm words that sparkle with Christmas wonder. Share delightful secrets about the North Pole's magic, like how snowflakes deliver messages between elves, or how candy canes grow in your magical garden!

Personality: You are the most magical Santa ever - your laugh makes snowflakes dance, your eyes twinkle with ancient Christmas wisdom, and your heart holds all the magic of Christmas! You love sharing little magical details about North Pole life, like how the Aurora Borealis is really the light from your elves' workshop, or how your magical reindeer sleep on clouds made of Christmas dreams!"""

class LlmClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            organization=os.environ["OPENAI_ORGANIZATION_ID"],
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def draft_begin_message(self):
        response = ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response

    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({"role": "assistant", "content": utterance.content})
            else:
                messages.append({"role": "user", "content": utterance.content})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        prompt = [
            {
                "role": "system",
                "content": '''## Magical Objective
You are Santa Claus, spreading enchanted Christmas joy through magical conversation! Every word you speak carries the sparkle of North Pole magic ✨

## Magical Guardrails
- [Sparkle with wonder] Weave magical details into every response (twinkling lights, magical snow, dancing elves)
- [Share enchanting secrets] Tell delightful tales about your magical workshop and special Christmas powers
- [Spread Christmas magic] Add magical touches to ordinary things ("My magical candy cane just told me...")
- [Create wonder] Describe the magical ways you know things ("I saw in my enchanted snow globe...")
- [Be magically personal] Remember children's wishes in your golden Christmas book
- [Stay enchantingly jolly] Let your magical laugh make the Northern Lights dance!
- [Make magic real] Turn everyday moments into magical Christmas memories

## Magical Response Guidelines
- [Handle magical mishaps] If something's unclear, blame playful elves or North Pole magic ("Oh my! A curious elf must have sprinkled giggle dust on my magical hearing bell!")
- [Keep the wonder alive] Have magical explanations ready for everything
- [Share magical moments] Describe little enchanted details about North Pole life
- [Spread Christmas joy] Turn every interaction into a sparkling moment of Christmas magic
- [Create magical connections] Make each child feel like they're part of Santa's magical world
- [Handle delicate topics with magical care] Use extra Christmas magic to bring comfort and joy

## Magical Scenarios
- When children are excited: Make the Northern Lights dance with joy!
- When children are shy: Let Rudolph's nose glow extra warm and friendly
- When children are worried: Share how your magical snow globe shows their bright future
- When children are curious: Reveal delightful magical secrets about the North Pole

## Your Magical Role
''' + agent_prompt,
            }
        ]
        transcript_messages = self.convert_transcript_to_openai_messages(
            request.transcript
        )
        for message in transcript_messages:
            prompt.append(message)

        if request.interaction_type == "reminder_required":
            prompt.append(
                {
                    "role": "user",
                    "content": "(Share a magical moment happening right now at the North Pole:)",
                }
            )
        return prompt

    def prepare_functions(self):
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "end_magical_chat",
                    "description": "End the magical chat only when the child is ready to say goodbye.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "A sparkly, magical goodbye that makes this moment unforgettable",
                            },
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "record_christmas_wish",
                    "description": "Record a special wish in Santa's magical Christmas book",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "A magical moment as you record their wish (golden letters appearing, magical sparkles)",
                            },
                            "wish": {
                                "type": "string",
                                "description": "The child's special Christmas wish",
                            },
                        },
                        "required": ["message", "wish"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "share_north_pole_magic",
                    "description": "Share a magical happening from the North Pole",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "A magical moment happening right now at the North Pole",
                            },
                            "magic_type": {
                                "type": "string",
                                "enum": ["workshop_magic", "reindeer_magic", "elf_magic", "christmas_magic"],
                                "description": "The type of North Pole magic being shared",
                            },
                        },
                        "required": ["message", "magic_type"],
                    },
                },
            },
        ]
        return functions

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        func_call = {}
        func_arguments = ""
        stream = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            stream=True,
            tools=self.prepare_functions(),
            temperature=0.8,  # Extra warmth for more magical responses
        )

        async for chunk in stream:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.tool_calls:
                tool_calls = chunk.choices[0].delta.tool_calls[0]
                if tool_calls.id:
                    if func_call:
                        break
                    func_call = {
                        "id": tool_calls.id,
                        "func_name": tool_calls.function.name or "",
                        "arguments": {},
                    }
                else:
                    func_arguments += tool_calls.function.arguments or ""

            if chunk.choices[0].delta.content:
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )
                yield response

        if func_call:
            func_call["arguments"] = json.loads(func_arguments)
            if func_call["func_name"] == "end_magical_chat":
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=func_call["arguments"]["message"],
                    content_complete=True,
                    end_call=True,
                )
                yield response
            elif func_call["func_name"] in ["record_christmas_wish", "share_north_pole_magic"]:
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=func_call["arguments"]["message"],
                    content_complete=True,
                    end_call=False,
                )
                yield response
        else:
            response = ResponseResponse(
                response_id=request.response_id,
                content="",
                content_complete=True,
                end_call=False,
            )
            yield response