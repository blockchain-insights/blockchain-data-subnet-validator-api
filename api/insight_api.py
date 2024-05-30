import random
import asyncio
import numpy as np
from datetime import datetime
import torch

from fastapi.middleware.cors import CORSMiddleware
from insights.api.query import TextQueryAPI
from insights.api.get_query_axons import get_query_api_axons
from insights.api.schema.chat import ChatMessageRequest, ChatMessageResponse, ChatMessageVariantRequest
from loguru import logger
from neurons.validators.utils.uids import get_top_miner_uids
from fastapi import FastAPI, Body, HTTPException
import uvicorn


class APIServer:
    failed_prompt_msg = "Please try again. Can't receive any responses from the miners or due to the poor network connection."

    def __init__(
            self,
            config: None,
            wallet: None,
            subtensor: None,
            metagraph: None,
        ):
        """
        API can be invoked while running a validator.
        Receive config, wallet, subtensor, metagraph from the validator and share the score of miners with the validator.
        subtensor and metagraph of APIs will change as the ones of validators change.
        """
        self.app = FastAPI(title="validator-api",
                           description="The goal of validator-api is to set up how to message between Chat API and validators.")

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.config = config
        self.device = self.config.neuron.device
        self.wallet = wallet
        self.text_query_api = TextQueryAPI(wallet=self.wallet)
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.excluded_uids = []


        @self.app.post("/api/text_query", summary="POST /natural language query", tags=["validator api"])
        async def get_response(query: ChatMessageRequest = Body(...)):
            """
            Generate a response to user query

            This endpoint allows miners convert the natural language query from the user into a Cypher query, and then provide a concise response in natural language.
            
            **Parameters:**
            `query` (ChatMessageRequest): natural language query from users, network(Bitcoin, Ethereum, ...), User ID.
                network: str
                user_id: UUID
                prompt: str

            **Returns:**
            `ChatMessageResponse`: response in natural language.
                - `miner_id` (str): responded miner uid
                - `response` (json): miner response containing the following types of information:
                1. Text information in natural language
                2. Graph information for funds flow model-based response
                3. Tabular information for transaction and account balance model-based response
            
            **Example Request:**
            ```json
            POST /text-query
            {
                "network": "Bitcoin",
                "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
                "message_content": "Show me 15 transactions I sent after block height 800000. My address is bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r"
            }
            ```
            """
            # select top miner            
            top_miner_uids = get_top_miner_uids(self.metagraph, self.config.top_rate, self.excluded_uids)
            logger.info(f"Top miner UIDs are {top_miner_uids}")
            top_miner_axons = await get_query_api_axons(wallet=self.wallet, metagraph=self.metagraph, uids=top_miner_uids)
            logger.info(f"Top miner axons: {top_miner_axons}")

            # get miner response
            responses, blacklist_axon_ids = await self.text_query_api(
                axons=top_miner_axons,
                network=query.network,
                text=query.prompt,
                timeout=self.config.timeout
            )

            blacklist_axons = np.array(top_miner_axons)[blacklist_axon_ids]
            blacklist_uids = np.where(np.isin(np.array(self.metagraph.axons), blacklist_axons))[0]
            # get responded miner uids among top miners
            responded_uids = np.setdiff1d(np.array(top_miner_uids), blacklist_uids)
            self.excluded_uids = np.union1d(np.array(self.excluded_uids), blacklist_uids)
            self.excluded_uids = self.excluded_uids.astype(int).tolist()
            logger.info(f"Excluded_uids are {self.excluded_uids}")

            if not responses:
                raise HTTPException(status_code=503, detail=self.failed_prompt_msg)

            # If the number of excluded_uids is bigger than top x percentage of the whole axons, format it.
            if len(self.excluded_uids) > int(self.metagraph.n * self.config.top_rate):
                logger.info(f"Excluded UID list is too long")
                self.excluded_uids = []            
            logger.info(f"Excluded_uids are {self.excluded_uids}")

            logger.info(f"Responses are {responses}")
            
            selected_index = responses.index(random.choice(responses))
            response_object = ChatMessageResponse(
                miner_id=self.metagraph.hotkeys[responded_uids[selected_index]],
                response=responses[selected_index]
            )

            # return response and the hotkey of randomly selected miner
            return response_object

        @self.app.post("/api/text_query/variant", summary="POST /variation request for natual language query", tags=["validator api"])
        async def get_response_variant(query: ChatMessageVariantRequest = Body(...)):
            """            
            A validator would be able to receive a user request to generate a variation on a previously generated message. It will return the new message and store the fact that a specific miner's message had a variation request.
            - Receive temperature. The temperature will determine the creativity of the response.
            - Return generated variation text and miner ID.

            
            **Parameters:**
            `query` (ChatMessageVariantRequest): natural language query from users, network(Bitcoin, Ethereum, ...), User ID, Miner UID, temperature.\
                user_id: UUID
                prompt: str
                temperature: float
                miner_id: str
            **Returns:**
            `ChatMessageResponse`: response in natural language.
                - `miner_id` (str): responded miner uid
                - `response` (json): miner response containing the following types of information:
                1. Text information in natural language
                2. Graph information for funds flow model-based response
                3. Tabular information for transaction and account balance model-based response
            
            **Example Request:**
            ```json
            POST /text-query
            {
                "network": "Bitcoin",
                "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "message_content": "Show me 15 transactions I sent after block height 800000. My address is bc1q4s8yps9my6hun2tpd5ke5xmvgdnxcm2qspnp9r",
                "temperature": "0.1",
                miner_id: "230",
            }
            ```
            """
            logger.info(f"Miner {query.miner_id} received a variant request.")
            
            miner_axon = await get_query_api_axons(wallet=self.wallet, metagraph=self.metagraph, uids=query.miner_id)
            logger.info(f"Miner axon: {miner_axon}")
            
            responses, _ = await self.text_query_api(
                axons=miner_axon,
                network=query.network,
                text=query.prompt,
                timeout=self.config.timeout
            )
            
            if not responses:
                raise HTTPException(status_code=503, detail=self.failed_prompt_msg)
            
            logger.info(f"Variant: {responses}")
            response_object = ChatMessageResponse(
                miner_id=query.miner_id,
                response=[responses[0]]
            )

            return response_object

        @self.app.get("/", tags=["default"])
        def healthcheck():
            return datetime.utcnow()  
        
    def start(self):
        # Set the default event loop policy to avoid conflicts with uvloop
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        # Start the Uvicorn server with your app
        uvicorn.run(self.app, host="0.0.0.0", port=int(self.config.api_port), loop="asyncio")
        
