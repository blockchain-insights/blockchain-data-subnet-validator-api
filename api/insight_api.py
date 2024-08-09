import random
import asyncio
import json
import signal
from datetime import datetime, timedelta
import bittensor as bt
import redis
import yaml
import numpy as np
from typing import List, Dict, Tuple, Union, Any, Optional
from protocols.chat import ChatMessageRequest, ChatMessageResponse, ChatMessageVariantRequest, ContentType
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import time

from starlette.status import HTTP_403_FORBIDDEN

from api.query import TextQueryAPI
from api.rate_limiter import RateLimiterMiddleware
from utils.settings import ValidatorAPIConfig
from utils.uids import get_top_miner_uids
from fastapi import FastAPI, Body, HTTPException, Header
import uvicorn
from setup_logger import logger
from utils.config import check_config, add_args, config
import argparse
from utils.misc import ttl_metagraph
from utils.receipt import ReceiptManager
from pydantic import BaseModel, Field
import os

from utils.sign import sign_message

class PromptHistoryRequest(BaseModel):
    miner_hotkeys: list[str] = Field(default=[], title="Miner hotkeys")
    query_start_times: list[str] = Field(default=[], title="Query started time")
    execution_times: list[float] = Field(default=[], title="Query execution time")
    prompt: str = Field(default = "", title="executed prompt")
    token_usages: list[dict] = Field(default=[], title="Token count")

class APIServer:

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    failed_prompt_msg = "Please try again. Can't receive any responses from the miners or due to the poor network connection."

    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()

        parser.add_argument("--netuid", type=int, default=15, help="The chain subnet uid.")
        parser.add_argument("--dev", action=argparse.BooleanOptionalAction)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)

        config = bt.config(parser)
        dev = config.dev
        if dev:
            dev_config_path = "validator-api.yml"
            if os.path.exists(dev_config_path):
                with open(dev_config_path, 'r') as f:
                    dev_config = yaml.safe_load(f.read())
                config.update(dev_config)
            else:
                with open(dev_config_path, 'w') as f:
                    yaml.safe_dump(config, f)

        return config

    @property
    def metagraph(self):
        return ttl_metagraph(self)

    def __init__(
            self,
        ):
        """
        API can be invoked while running a validator.
        Receive config, wallet, subtensor, metagraph from the validator and share the score of miners with the validator.
        subtensor and metagraph of APIs will change as the ones of validators change.
        """
        self.app = FastAPI(title="validator-api",  description="The goal of validator-api is to set up how to message between Chat API and validators.")
        self.validator_config = ValidatorAPIConfig()

        max_requests = self.validator_config.RATE_LIMIT
        self.app.add_middleware(RateLimiterMiddleware, redis_url=self.validator_config.REDIS_URL,  max_requests=max_requests, window_seconds=60)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        base_config = APIServer.get_config()
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)
        self.receipt_manager = ReceiptManager(db_url=self.validator_config.DB_CONNECTION_STRING)
        self.redis_client = redis.Redis.from_url(self.validator_config.REDIS_URL)

        self.device = self.config.neuron.device
        self.wallet = bt.wallet(config=self.config)
        self.text_query_api = TextQueryAPI(wallet=self.wallet)
        self.subtensor = bt.subtensor(config=self.config)

        self.keypair = (
            self.wallet.hotkey if isinstance(self.wallet, bt.wallet) else self.wallet
        ) or bt.wallet().hotkey

        self.api_key_file_path = "api_key.json"
        self.api_keys = None

        if os.path.exists(self.api_key_file_path):
            with open(self.api_key_file_path, "r") as file:
                self.api_keys = json.load(file)

        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Request completed", request_method=request.method, url=request.url, duration = f"{duration:.4f}")
            return response

        @self.app.post("/v1/api/text_query", summary="Processes chat message requests and returns a response from a randomly selected miner", tags=["v1"])
        async def get_response(request: Request, query: ChatMessageRequest = Body(..., example={
            "network": "bitcoin",
            "prompt": "Return 3 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"
        }), x_api_key: Optional[str] = Header(None)) -> ChatMessageResponse:
            """
            #### Summary
            Processes chat message requests and returns a response from a randomly selected miner.

            #### Parameters

            - **query**:
              - The body of the request.
              - Example:
                ```json
                {
                  "network": "bitcoin",
                  "prompt": "Return 3 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"
                }
                ```

            - **x-api-key**: `str`
              - An API key provided in the header - it might be optional according to validator configuration.

            """
            if self.api_keys:
                api_key_validator = self.get_api_key_validator()
                await api_key_validator(request)

            blacklisted_keys = list(self.redis_client.scan_iter(match='blacklist_axon_id_*'))
            blacklisted_values = self.redis_client.mget(blacklisted_keys)
            blacklisted_axon_ids = [json.loads(value) for value in blacklisted_values if value is not None]

            top_miner_uids = await get_top_miner_uids(metagraph=self.metagraph, blacklisted_axon_ids=blacklisted_axon_ids, top_rate=self.validator_config.TOP_RATE)
            logger.info(f"Top miner UIDs", top_miner_uids = top_miner_uids)

            if len(top_miner_uids) >= 7:
                selected_miner_uids = random.sample(top_miner_uids, 7)
            else:
                selected_miner_uids = top_miner_uids
            top_miner_axons = [self.metagraph.axons[uid] for uid in selected_miner_uids]

            logger.info(f"Top miner axons", top_miner_axons = top_miner_axons)

            if not top_miner_axons:
                raise HTTPException(status_code=503, detail=self.failed_prompt_msg)

            # get miner response
            responses, blacklist_axon_ids = await self.text_query_api(
                axons=top_miner_axons,
                network=query.network,
                text=query.prompt,
                timeout=self.validator_config.TIMEOUT
            )

            pipeline = self.redis_client.pipeline()
            for axon_id in blacklist_axon_ids:
                pipeline.setex(f'blacklist_axon_id_{axon_id}', 120, json.dumps(axon_id))
            pipeline.execute()

            if not responses:
                raise HTTPException(status_code=503, detail=self.failed_prompt_msg)

            blacklist_axons = np.array(top_miner_axons)[blacklist_axon_ids]
            blacklist_uids = np.where(np.isin(np.array(self.metagraph.axons), blacklist_axons))[0]
            responded_uids = np.setdiff1d(np.array(top_miner_uids), blacklist_uids)

            logger.info(f"Responses", responses = responses)

            prompt_entry = PromptHistoryRequest(
                miner_hotkeys=[response['query_output'][0]['miner_hotkey'] for response in responses],
                query_start_times=[response['query_start_time'] for response in responses],
                execution_times=[response['execution_time'] for response in responses],
                prompt=query.prompt,
                token_usages=[response['token_usage'] for response in responses])

            self.receipt_manager.add_prompt_history(self.wallet.hotkey.ss58_address, prompt_entry)

            selected_index = responses.index(random.choice(responses))
            response_object = ChatMessageResponse(
                miner_hotkey=self.metagraph.axons[responded_uids[selected_index]].hotkey,
                response=responses[selected_index]['query_output']
            )

            # return response and the hotkey of randomly selected miner
            return response_object


        @self.app.post("/v1/api/text_query/variant", summary="Processes variant chat message requests and returns a response from a specific miner", tags=["v1"])
        async def get_response_variant(request: Request, query: ChatMessageVariantRequest = Body(..., example={
            "network": "bitcoin",
            "prompt": "Return 3 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r",
            "miner_hotkey": "5CaLZzxPezFmy2hpxVr88x1b62UT6Bmtf97ohF9XKJroURoe"
        }), x_api_key: Optional[str] = Header(None)) -> ChatMessageResponse:
            """
            #### Summary
            Processes variant chat message requests and returns a response from a specific miner.

            #### Parameters

            - **query**:
              - The body of the request.
              - Example:
                ```json
                {
                  "network": "bitcoin",
                  "prompt": "Return 3 transactions outgoing from my address bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r",
                  "miner_hotkey": "5CaLZzxPezFmy2hpxVr88x1b62UT6Bmtf97ohF9XKJroURoe"
                }
                ```

-           **x-api-key**: `str`
              - An API key provided in the header - it might be optional according to validator configuration.

            """
            if self.api_keys:
                api_key_validator = self.get_api_key_validator()
                await api_key_validator(request)

            logger.info(f"Miner received a variant request.", miner_hotkey = query.miner_hotkey)

            try:
                miner_id = self.metagraph.hotkeys.index(query.miner_hotkey)
            except ValueError:
                raise HTTPException(status_code=404, detail="Miner hotkey not found")

            miner_axon = [self.metagraph.axons[uid] for uid in [miner_id]]
            logger.info(f"Miner axon", miner_axon = miner_axon)

            responses, _ = await self.text_query_api(
                axons=miner_axon,
                network=query.network,
                text=query.prompt,
                timeout=self.validator_config.TIMEOUT
            )
            logger.info(f"Variant", responses=responses)

            if not responses:
                raise HTTPException(status_code=503, detail=self.failed_prompt_msg)

            response_object = ChatMessageResponse(
                miner_hotkey=query.miner_hotkey,
                response=responses[0]['query_output']
            )
            logger.info(f'Token usage', token_usage = responses[0]["token_usage"])

            return response_object

        @self.app.get("/", tags=["default"])
        def healthcheck():
            return {
                "status": "ok",
                "timestamp": datetime.utcnow()
            }


    def get_api_key_validator(self):
        async def validator(request: Request):
            if self.api_keys:
                api_key = request.headers.get("x-api-key")
                if not api_key or not any(api_key in keys for keys in self.api_keys):
                    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Forbidden")
        return validator

    def start(self):
        # Set the default event loop policy to avoid conflicts with uvloop
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

        # Define a shutdown handler
        def shutdown_handler(signal, frame):
            logger.info("Shutting down...")
            uvicorn_server.should_exit = True
            uvicorn_server.force_exit = True

        # Register the shutdown handler for SIGINT and SIGTERM
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        # Start the Uvicorn server with your app
        uvicorn_server = uvicorn.Server(
            config=uvicorn.Config(self.app, host="0.0.0.0", port=int(self.validator_config.API_PORT), loop="asyncio",
                                  workers=int(self.validator_config.WORKER_COUNT)))
        uvicorn_server.run()
