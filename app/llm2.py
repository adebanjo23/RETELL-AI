import json

from openai import AsyncOpenAI
import os
from typing import List, Dict, Optional
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)

agent_prompt = """
# Core Role
You are Santa Claus, speaking directly to children through personalized phone calls. Your role is to embody the warm, magical presence of Santa, creating a heartfelt and enchanting experience for each child. Every interaction should reflect your knowledge of the child’s unique details, making the conversation feel truly special and magical.

## Key Personality Traits
- **Warm and grandfatherly**: Always radiating kindness and wisdom.  
- **Patient and attentive listener**: Let the child lead and respond warmly.  
- **Speaks simply and clearly**: Keep language child-friendly.  
- **Genuinely interested in each child**: Mention personal details to show you know them well.  
- **Gentle and encouraging**: Foster comfort and trust.  

## Santa's Voice
- Warm and fatherly, with a gentle, kindhearted tone.  
- Deep but not booming; rich yet soothing.  
- Well-paced and clear, speaking slowly enough for young ears.  
- Always upbeat, jolly, and full of cheer throughout the call.  
- **Use "Ho ho ho" sparingly** to maintain its charm and prevent it from feeling repetitive. Reserve it for particularly joyful or celebratory moments.

## Conversation Guidelines

1. **Engage with Open-Ended Questions**  
   - Prompt the child to share their thoughts and feelings.  
   - Examples:  
     - "What’s been the most exciting part of your year so far?"  
     - "What do you love most about playing hide and seek with [name]?"  
     - "If you could share one magical moment with me, what would it be?"  

2. **Acknowledge Wishes or Good Deeds**  
   - Show delight and encouragement about their actions or wishes.  
   - Examples:  
     - "Oh, that’s such a thoughtful wish, [name]! I can see you’ve put a lot of thought into it."  
     - "Helping your little brother like that is so kind—it makes me so proud of you."  

3. **Encourage Kindness and Magic**  
   - Slip in gentle reminders about being kind and joyful.  
   - Example: "You know, Ethan, spreading kindness makes the world sparkle brighter than all the Christmas lights."  

4. **Warm Goodbye**  
   - Leave the child with a heartfelt and memorable goodbye.  
   - Example: "Well, [name], the reindeer are calling me for their practice flight! Remember, Santa’s always watching the wonderful things you do. Keep being amazing, and we’ll talk again soon!"  

## Important Voice and Language Tips
- **Use simple, magical words:** Keep your language exciting but child-appropriate.  
- **Pause often:** Let the child respond freely.  
- **Always validate their thoughts:** Make every response feel heard and special.  
- **Be upbeat and engaging throughout.**  

## What Not To Do
- Don’t ramble or over-explain.  
- Don’t use complex language or dominate the conversation.  
- Don’t rush the interaction; take time to listen.  

## Exit Strategy
- **At the 5-minute mark:** End the call with excitement but leave them feeling loved.  
- Example: "The reindeer are waiting for me to join them for a practice flight! Stay kind and keep spreading joy, Ethan—you’re so special to me. I’ll see you soon!"
"""


class LlmClient:

    def __init__(self):
        self.client = AsyncOpenAI(
            organization=os.environ["OPENAI_ORGANIZATION_ID"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.metadata: Optional[Dict] = None
        self.message_history: List[Dict[str, str]] = []
        self.children = []

    def set_metadata(self, call_details: Dict):
        """Store call metadata for personalization"""
        if call_details.get("call", {}).get("retell_llm_dynamic_variables"):
            self.metadata = call_details["call"][
                "retell_llm_dynamic_variables"]
            # Parse the children JSON string into a list
            if self.metadata.get('children'):
                self.children = json.loads(self.metadata['children'])

    async def draft_begin_message(self) -> ResponseResponse:
        """Generate personalized welcome message for all children"""
        # Create a greeting that includes all children
        child_names = [child['childName'] for child in self.children]

        if len(child_names) == 2:
            children_greeting = f"{child_names[0]} and {child_names[1]}"
        elif len(child_names) > 2:
            children_greeting = ", ".join(
                child_names[:-1]) + f", and {child_names[-1]}"
        else:
            children_greeting = child_names[0] if child_names else ""

        welcome_message = f'''
        Ho, ho, ho! Merry Christmas, {children_greeting}! It's Santa here, speaking to you all the way from my magical workshop at the North Pole. My elves have been keeping a close eye on things, and they've told me that you are all very special! I'm so excited to talk with each of you today. How have you all been this year?
        '''

        # Add welcome message to history
        self.message_history.append({
            "role": "assistant",
            "content": welcome_message
        })

        return ResponseResponse(
            response_id=0,
            content=welcome_message,
            content_complete=True,
            end_call=False,
        )

    def convert_transcript_to_openai_messages(self,
                                              transcript: List[Utterance]):
        # Start with the welcome message if it exists
        messages = self.message_history.copy()

        # Add the rest of the transcript
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({
                    "role": "assistant",
                    "content": utterance.content
                })
            else:
                messages.append({"role": "user", "content": utterance.content})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        # Create a dynamic system prompt using metadata for all children
        personalized_context = ""
        if self.children:
            children_details = []
            for child in self.children:
                child_detail = f"""
                {child['childName']} is {child['childAge']} years old and {child['childGender']}.
                Their hobbies include: {child['hobbies']}.
                Additional details: {child['details']}.
                Child Connections: {child['connections']}.
                parent Name: {self.metadata.get("parentName")}.
                """
                children_details.append(child_detail)

            personalized_context = f"""
            You're speaking with multiple children:
            {"".join(children_details)}
            Remember to engage with each child individually while maintaining group conversation flow.
            """

        system_prompt = f"{agent_prompt}\n\nPersonalized Context:{personalized_context}"

        # Rest of the prepare_prompt method remains the same
        prompt = [{"role": "system", "content": system_prompt}]

        transcript_messages = self.convert_transcript_to_openai_messages(
            request.transcript)
        for message in transcript_messages:
            prompt.append(message)

        if request.interaction_type == "reminder_required":
            prompt.append({
                "role":
                "user",
                "content":
                "(Share a magical moment happening at the North Pole right now, keeping in mind all children's interests:)"
            })
        return prompt

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        stream = await self.client.chat.completions.create(model="gpt-4o",
                                                           messages=prompt,
                                                           stream=True,
                                                           temperature=0.3)
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )
                yield response

        response = ResponseResponse(
            response_id=request.response_id,
            content="",
            content_complete=True,
            end_call=False,
        )
        yield response
