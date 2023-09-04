import os
import time
from typing import List
from uuid import UUID
from venv import logger

from auth import AuthBearer, get_current_user
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from llm.openai import OpenAIBrainPicking
from llm.qa_headless import HeadlessQA
from models import (
    Brain,
    BrainEntity,
    Chat,
    ChatQuestion,
    UserIdentity,
    UserUsage,
    get_supabase_db,
)
from models.databases.supabase.supabase import SupabaseDB
from repository.brain import get_brain_details
from repository.chat import (
    ChatUpdatableProperties,
    CreateChatProperties,
    GetChatHistoryOutput,
    create_chat,
    get_chat_by_id,
    get_chat_history,
    get_user_chats,
    update_chat,
)
from repository.user_identity import get_user_identity

chat_router = APIRouter()


class NullableUUID(UUID):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v) -> UUID | None:
        if v == "":
            return None
        try:
            return UUID(v)
        except ValueError:
            return None


def delete_chat_from_db(supabase_db: SupabaseDB, chat_id):
    try:
        supabase_db.delete_chat_history(chat_id)
    except Exception as e:
        print(e)
        pass
    try:
        supabase_db.delete_chat(chat_id)
    except Exception as e:
        print(e)
        pass


def check_user_requests_limit(
    user: UserIdentity,
):
    userDailyUsage = UserUsage(
        id=user.id, email=user.email, openai_api_key=user.openai_api_key
    )

    date = time.strftime("%Y%m%d")
    userDailyUsage.handle_increment_user_request_count(date)

    if user.openai_api_key is None:
        max_requests_number = int(os.getenv("MAX_REQUESTS_NUMBER", 1))
        if int(userDailyUsage.daily_requests_count) >= int(max_requests_number):
            raise HTTPException(
                status_code=429,  # pyright: ignore reportPrivateUsage=none
                detail="You have reached the maximum number of requests for today.",  # pyright: ignore reportPrivateUsage=none
            )
    else:
        pass


@chat_router.get("/chat/healthz", tags=["Health"])
async def healthz():
    return {"status": "ok"}


# get all chats
@chat_router.get("/chat", dependencies=[Depends(AuthBearer())], tags=["Chat"])
async def get_chats(current_user: UserIdentity = Depends(get_current_user)):
    """
    Retrieve all chats for the current user.

    - `current_user`: The current authenticated user.
    - Returns a list of all chats for the user.

    This endpoint retrieves all the chats associated with the current authenticated user. It returns a list of chat objects
    containing the chat ID and chat name for each chat.
    """
    chats = get_user_chats(str(current_user.id))
    return {"chats": chats}


# delete one chat
@chat_router.delete(
    "/chat/{chat_id}", dependencies=[Depends(AuthBearer())], tags=["Chat"]
)
async def delete_chat(chat_id: UUID):
    """
    Delete a specific chat by chat ID.
    """
    supabase_db = get_supabase_db()
    delete_chat_from_db(supabase_db=supabase_db, chat_id=chat_id)
    return {"message": f"{chat_id}  has been deleted."}


# update existing chat metadata
@chat_router.put(
    "/chat/{chat_id}/metadata", dependencies=[Depends(AuthBearer())], tags=["Chat"]
)
async def update_chat_metadata_handler(
    chat_data: ChatUpdatableProperties,
    chat_id: UUID,
    current_user: UserIdentity = Depends(get_current_user),
) -> Chat:
    """
    Update chat attributes
    """

    chat = get_chat_by_id(chat_id)  # pyright: ignore reportPrivateUsage=none
    if str(current_user.id) != chat.user_id:
        raise HTTPException(
            status_code=403,  # pyright: ignore reportPrivateUsage=none
            detail="You should be the owner of the chat to update it.",  # pyright: ignore reportPrivateUsage=none
        )
    return update_chat(chat_id=chat_id, chat_data=chat_data)


# create new chat
@chat_router.post("/chat", dependencies=[Depends(AuthBearer())], tags=["Chat"])
async def create_chat_handler(
    chat_data: CreateChatProperties,
    current_user: UserIdentity = Depends(get_current_user),
):
    """
    Create a new chat with initial chat messages.
    """

    return create_chat(user_id=current_user.id, chat_data=chat_data)


# add new question to chat
@chat_router.post(
    "/chat/{chat_id}/question",
    dependencies=[
        Depends(
            AuthBearer(),
        ),
    ],
    tags=["Chat"],
)
async def create_question_handler(
    request: Request,
    chat_question: ChatQuestion,
    chat_id: UUID,
    brain_id: NullableUUID
    | UUID
    | None = Query(..., description="The ID of the brain"),
    current_user: UserIdentity = Depends(get_current_user),
) -> GetChatHistoryOutput:
    """
    Add a new question to the chat.
    """
    # Retrieve user's OpenAI API key
    current_user.openai_api_key = request.headers.get("Openai-Api-Key")
    brain = Brain(id=brain_id)

    if not current_user.openai_api_key and brain_id:
        brain_details = get_brain_details(brain_id)
        if brain_details:
            current_user.openai_api_key = brain_details.openai_api_key

    if not current_user.openai_api_key:
        user_identity = get_user_identity(current_user.id)

        if user_identity is not None:
            current_user.openai_api_key = user_identity.openai_api_key

    # Retrieve chat model (temperature, max_tokens, model)
    if (
        not chat_question.model
        or not chat_question.temperature
        or not chat_question.max_tokens
    ):
        # TODO: create ChatConfig class (pick config from brain or user or chat) and use it here
        chat_question.model = chat_question.model or brain.model or "gpt-3.5-turbo"
        chat_question.temperature = chat_question.temperature or brain.temperature or 0
        chat_question.max_tokens = chat_question.max_tokens or brain.max_tokens or 256

    try:
        check_user_requests_limit(current_user)

        gpt_answer_generator: HeadlessQA | OpenAIBrainPicking
        if brain_id:
            gpt_answer_generator = OpenAIBrainPicking(
                chat_id=str(chat_id),
                model=chat_question.model,
                max_tokens=chat_question.max_tokens,
                temperature=chat_question.temperature,
                brain_id=str(brain_id),
                user_openai_api_key=current_user.openai_api_key,  # pyright: ignore reportPrivateUsage=none
                prompt_id=chat_question.prompt_id,
            )
        else:
            gpt_answer_generator = HeadlessQA(
                model=chat_question.model,
                temperature=chat_question.temperature,
                max_tokens=chat_question.max_tokens,
                user_openai_api_key=current_user.openai_api_key,
                chat_id=str(chat_id),
                prompt_id=chat_question.prompt_id,
            )

        chat_answer = gpt_answer_generator.generate_answer(chat_id, chat_question)

        return chat_answer
    except HTTPException as e:
        raise e


# stream new question response from chat
@chat_router.post(
    "/chat/{chat_id}/question/stream",
    dependencies=[
        Depends(
            AuthBearer(),
        ),
    ],
    tags=["Chat"],
)
async def create_stream_question_handler(
    request: Request,
    chat_question: ChatQuestion,
    chat_id: UUID,
    brain_id: NullableUUID
    | UUID
    | None = Query(..., description="The ID of the brain"),
    current_user: UserIdentity = Depends(get_current_user),
) -> StreamingResponse:
    # TODO: check if the user has access to the brain

    # Retrieve user's OpenAI API key
    current_user.openai_api_key = request.headers.get("Openai-Api-Key")
    brain = Brain(id=brain_id)
    brain_details: BrainEntity | None = None
    if not current_user.openai_api_key and brain_id:
        brain_details = get_brain_details(brain_id)
        if brain_details:
            current_user.openai_api_key = brain_details.openai_api_key

    if not current_user.openai_api_key:
        user_identity = get_user_identity(current_user.id)

        if user_identity is not None:
            current_user.openai_api_key = user_identity.openai_api_key

    # Retrieve chat model (temperature, max_tokens, model)
    if (
        not chat_question.model
        or chat_question.temperature is None
        or not chat_question.max_tokens
    ):
        # TODO: create ChatConfig class (pick config from brain or user or chat) and use it here
        chat_question.model = chat_question.model or brain.model or "gpt-3.5-turbo"
        chat_question.temperature = chat_question.temperature or brain.temperature or 0
        chat_question.max_tokens = chat_question.max_tokens or brain.max_tokens or 256

    try:
        logger.info(f"Streaming request for {chat_question.model}")
        check_user_requests_limit(current_user)
        gpt_answer_generator: HeadlessQA | OpenAIBrainPicking
        if brain_id:
            gpt_answer_generator = OpenAIBrainPicking(
                chat_id=str(chat_id),
                model=(brain_details or chat_question).model
                if current_user.openai_api_key
                else "gpt-3.5-turbo",  # type: ignore
                max_tokens=(brain_details or chat_question).max_tokens
                if current_user.openai_api_key
                else 0,  # type: ignore
                temperature=(brain_details or chat_question).temperature
                if current_user.openai_api_key
                else 256,  # type: ignore
                brain_id=str(brain_id),
                user_openai_api_key=current_user.openai_api_key,  # pyright: ignore reportPrivateUsage=none
                streaming=True,
                prompt_id=chat_question.prompt_id,
            )
        else:
            gpt_answer_generator = HeadlessQA(
                model=chat_question.model
                if current_user.openai_api_key
                else "gpt-3.5-turbo",
                temperature=chat_question.temperature
                if current_user.openai_api_key
                else 256,
                max_tokens=chat_question.max_tokens
                if current_user.openai_api_key
                else 0,
                user_openai_api_key=current_user.openai_api_key,  # pyright: ignore reportPrivateUsage=none
                chat_id=str(chat_id),
                streaming=True,
                prompt_id=chat_question.prompt_id,
            )

        print("streaming")
        return StreamingResponse(
            gpt_answer_generator.generate_stream(chat_id, chat_question),
            media_type="text/event-stream",
        )

    except HTTPException as e:
        raise e


# get chat history
@chat_router.get(
    "/chat/{chat_id}/history", dependencies=[Depends(AuthBearer())], tags=["Chat"]
)
async def get_chat_history_handler(
    chat_id: UUID,
) -> List[GetChatHistoryOutput]:
    # TODO: RBAC with current_user
    return get_chat_history(str(chat_id))
