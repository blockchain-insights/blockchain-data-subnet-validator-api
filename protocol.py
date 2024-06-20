from typing import Optional, List, Dict
import bittensor as bt
from protocols.llm_engine import LlmMessage, QueryOutput
from pydantic import BaseModel

# protocol version
VERSION = 5
ERROR_TYPE = int
MAX_MINER_INSTANCE = 9


class DiscoveryMetadata(BaseModel):
    network: str = None


class DiscoveryOutput(BaseModel):
    metadata: DiscoveryMetadata = None
    block_height: int = None
    start_block_height: int = None
    balance_model_last_block: int = None
    run_id: str = None
    version: Optional[int] = VERSION


class BaseSynapse(bt.Synapse):
    version: int = VERSION


class HealthCheck(BaseSynapse):
    output: Optional[List[Dict]] = None

    def deserialize(self):
        return self.output

class LlmQuery(BaseSynapse):
    network: str = None    
    # decide whether to invoke a generic llm endpoint or not
    # is_generic_llm: bool = False  
    # messages: conversation history for llm agent to use as context
    messages: List[LlmMessage] = None

    # output
    output: Optional[List[QueryOutput]] = None

    def deserialize(self) -> str:
        return self.output