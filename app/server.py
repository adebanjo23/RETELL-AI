import json
import os
import asyncio
from retell import Retell

import requests
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from concurrent.futures import TimeoutError as ConnectionTimeoutError
from .custom_types import (
    ConfigResponse,
    ResponseRequiredRequest,
)
from .llm import LlmClient  # or use .llm_with_func_calling

clientRetell = Retell(api_key=os.getenv("RETELL_API_KEY"))

load_dotenv(override=True)
app = FastAPI()
retell = Retell(api_key=os.environ["RETELL_API_KEY"])


async def send_call_analyzed_webhook(call_data: Dict[Any, Any]) -> bool:
    WEBHOOK_URL = "https://services.leadconnectorhq.com/hooks/jyPDXTf3YpjI9G74bRCW/webhook-trigger/3QRbbyY4pTiNh10dSZ68"

    headers = {
        'Content-Type': 'application/json',
    }

    try:
        print(f"recording_url: {call_data.recording_url}")
        if call_data.retell_llm_dynamic_variables.get('contact_recording') == "true":
            print(f"recording_url: {call_data.recording_url}")
            webhook_payload = {
                "to_number": call_data.to_number,
                "recording_url": call_data.recording_url
            }
            # Send the webhook
            response = requests.post(
                WEBHOOK_URL,
                headers=headers,
                json=webhook_payload,
                timeout=10  # 10 second timeout
            )

            # Check if the request was successful
            response.raise_for_status()

            print(
                "Successfully sent call_analyzed webhook"
            )
            return True
        else:
            print("No recording")

    except requests.exceptions.RequestException as e:
        error_message = f"Failed to send webhook: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        error_message = f"Unexpected error sending webhook: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)


# Handle webhook from Retell server. This is used to receive events from Retell server.
# Including call_started, call_ended, call_analyzed
@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        post_data = await request.json()
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":"), ensure_ascii=False),
            api_key=str(os.environ["RETELL_API_KEY"]),
            signature=str(request.headers.get("X-Retell-Signature")),
        )
        if not valid_signature:
            print(
                "Received Unauthorized",
                post_data["event"],
                post_data["data"]["call_id"],
            )
            return JSONResponse(status_code=401,
                                content={"message": "Unauthorized"})
        if post_data["event"] == "call_started":
            print("Call started event", post_data["data"]["call_id"])
        elif post_data["event"] == "call_ended":
            print("Call ended event", post_data["data"]["call_id"])
        elif post_data["event"] == "call_analyzed":
            print("Call analyzed event", post_data["data"]["call_id"])
            await send_call_analyzed_webhook(post_data["data"])

        else:
            print("Unknown event", post_data["event"])
        return JSONResponse(status_code=200, content={"received": True})
    except Exception as err:
        print(f"Error in webhook: {err}")
        return JSONResponse(status_code=500,
                            content={"message": "Internal Server Error"})


# Start a websocket server to exchange text input and output with Retell server. Retell server
# will send over transcriptions and other information. This server here will be responsible for
# generating responses with LLM and send back to Retell server.
@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    try:
        await websocket.accept()
        llm_client = LlmClient()
        call_details_received = False

        # Send optional config to Retell server
        config = ConfigResponse(
            response_type="config",
            config={
                "auto_reconnect": True,
                "call_details": True,
            },
            response_id=1,
        )
        await websocket.send_json(config.__dict__)

        response_id = 0

        async def handle_message(request_json):
            nonlocal response_id, call_details_received

            if request_json["interaction_type"] == "call_details":
                print("Call details received:",
                      json.dumps(request_json, indent=2))
                llm_client.set_metadata(request_json)
                call_details_received = True

                # Generate and send first message only after receiving call details
                first_event = await llm_client.draft_begin_message()
                await websocket.send_json(first_event.__dict__)
                return

            if request_json["interaction_type"] == "ping_pong":
                await websocket.send_json({
                    "response_type":
                        "ping_pong",
                    "timestamp":
                        request_json["timestamp"],
                })
                return

            if request_json["interaction_type"] == "update_only":
                return

            if (request_json["interaction_type"] == "response_required" or
                    request_json["interaction_type"] == "reminder_required"):
                response_id = request_json["response_id"]
                request = ResponseRequiredRequest(
                    interaction_type=request_json["interaction_type"],
                    response_id=response_id,
                    transcript=request_json["transcript"],
                )
                print(
                    f"""Received interaction_type={request_json['interaction_type']}, response_id={response_id}, last_transcript={request_json['transcript'][-1]['content']}"""
                )

                async for event in llm_client.draft_response(request):
                    await websocket.send_json(event.__dict__)
                    if request.response_id < response_id:
                        break  # new response needed, abandon this one

        async for data in websocket.iter_json():
            asyncio.create_task(handle_message(data))

    except WebSocketDisconnect:
        print(f"LLM WebSocket disconnected for {call_id}")
    except ConnectionTimeoutError as e:
        print(f"Connection timeout error for {call_id}")
    except Exception as e:
        print(f"Error in LLM WebSocket: {e} for {call_id}")
        await websocket.close(1011, "Server error")
    finally:
        print(f"LLM WebSocket connection closed for {call_id}")
        call_response = retell.call.retrieve(call_id)
        print(call_response)
        # if call_response.status_code == 200:
        await send_call_analyzed_webhook(call_response)
