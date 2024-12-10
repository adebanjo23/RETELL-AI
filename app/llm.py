from time import time

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
- **Do not Use "Ho ho ho" during the call

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
- Do not say Ho-ho-ho, during the call.

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
        self.message_history: List[Dict[str, str]] = []  # Track all messages
        self.start_time: float = time()

    def set_metadata(self, call_details: Dict):
        """Store call metadata for personalization"""
        if call_details.get("call", {}).get("retell_llm_dynamic_variables"):
            self.metadata = call_details["call"][
                "retell_llm_dynamic_variables"]

    async def draft_begin_message(self) -> ResponseResponse:
        """Generate personalized welcome message using metadata"""
        if not self.metadata:
            fallback_message = '''
            "Ho-Ho-Ho, Merry Christmas!! It’s Santa here, speaking to you all the way from my magical workshop at the North Pole. My elves have been keeping a close eye on things, and they’ve told me that you’re someone very special! I’m so excited to talk with you. What is your name my dear?"
            '''
            self.message_history.append({
                "role": "assistant",
                "content": fallback_message
            })
            return ResponseResponse(
                response_id=0,
                content=fallback_message,
                content_complete=True,
                end_call=False,
            )

        welcome_message = f'''
        Ho-Ho-Ho Merry Christmas, {self.metadata.get('contact_child_name', '')}! It’s Santa here, speaking to you all the way from my magical workshop at the North Pole. My elves have been keeping a close eye on things, and they’ve told me that you’re someone very special! I’m so excited to talk with you. Tell me, how have you been this year?
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
        personalized_context = ""
        if self.metadata:
            personalized_context = f"""
            You're speaking with {self.metadata.get('contact_child_name', 'a child')},
            Who is {self.metadata.get('contact_child_age', '')} years old,
            Their hobbies include: {self.metadata.get('contact_hobbies', '')}.
            Additional details: {self.metadata.get('contact_additional_information', '')}.
            Child Connections: {self.metadata.get('contact_family_info', '')}.  
            """

        system_prompt = f"{agent_prompt}\n\nPersonalized Context:{personalized_context}"

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
                "(Share a magical moment happening at the North Pole right now, keeping in mind the child's interests:)"
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
