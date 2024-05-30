# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 aph5nt
import concurrent
import threading
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import argparse
import traceback

import torch
import bittensor as bt
import os
import yaml
import json

import insights
from insights.api.insight_api import APIServer
from insights.protocol import Discovery, DiscoveryOutput, MAX_MINER_INSTANCE, MODEL_TYPE_BALANCE_TRACKING, \
    NETWORK_BITCOIN

from neurons.remote_config import ValidatorConfig
from neurons.nodes.factory import NodeFactory
from neurons.storage import store_validator_metadata
from neurons.validators.benchmark import BenchmarkValidator
from neurons.validators.challenge_factory.balance_challenge_factory import BalanceChallengeFactory
from neurons.validators.scoring import Scorer
from neurons.validators.uptime import MinerUptimeManager
from neurons.validators.utils.metadata import Metadata
from neurons.validators.utils.ping import ping
from neurons.validators.utils.synapse import is_discovery_response_valid
from neurons.validators.utils.uids import get_uids_batch
from template.base.validator import BaseValidatorNeuron
from neurons import logger


class Validator(BaseValidatorNeuron):

    @staticmethod
    def get_config():

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--alpha", default=0.9, type=float, help="The weight moving average scoring.py."
        )

        parser.add_argument("--netuid", type=int, default=15, help="The chain subnet uid.")
        parser.add_argument("--dev", action=argparse.BooleanOptionalAction)

        # For API configuration
        # Subnet Validator and validator API
        # You can invoke the API while instantiating the validator.
        # To run API, it's needed to set `enable_api`, `api_port`, `top_rate`, `timeout`, `user_query_moving_average_alpha` additionally.

        parser.add_argument("--enable_api", type=bool, default=False, help="Decide whether to launch api or not.")
        parser.add_argument("--api_port", type=int, default=8001, help="API endpoint port.")
        parser.add_argument("--timeout", type=int, default=360, help="Timeout.")
        parser.add_argument("--top_rate", type=float, default=0.80, help="Best selection percentage")
        parser.add_argument("--user_query_moving_average_alpha", type=float, default=0.0001,
                            help="Moving average alpha for scoring user query miners.")

        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)

        config = bt.config(parser)
        config.db_connection_string = os.environ.get('DB_CONNECTION_STRING', '')

        dev = config.dev
        if dev:
            dev_config_path = "validator.yml"
            if os.path.exists(dev_config_path):
                with open(dev_config_path, 'r') as f:
                    dev_config = yaml.safe_load(f.read())
                config.update(dev_config)
            else:
                with open(dev_config_path, 'w') as f:
                    yaml.safe_dump(config, f)

        def _copy(newconfig, config, allow):
            if (isinstance(allow, str)):
                newconfig[allow] = config[allow]
            elif (isinstance(allow, tuple)):
                if (len(allow) == 1):
                    newconfig[allow[0]] = config[allow[0]]
                else:
                    if (newconfig.get(allow[0]) == None): newconfig[allow[0]] = {}
                    _copy(newconfig[allow[0]], config[allow[0]], allow[1:])

        def filter(config, allowlist):
            newconfig = {}
            for item in allowlist:
                _copy(newconfig, config, item)
            return newconfig

        whitelist_config_keys = {'alpha', 'api_port', ('logging', 'logging_dir'), ('logging', 'record_log'), 'netuid',
                                 ('subtensor', 'chain_endpoint'), ('subtensor', 'network'), 'timeout', 'top_rate',
                                 'wallet'}

        json_config = json.loads(json.dumps(config, indent=2))
        config_out = filter(json_config, whitelist_config_keys)
        logger.info('config', config=config_out)

        return config

    def __init__(self, config=None):
        config = Validator.get_config()
        self.validator_config = ValidatorConfig().load_and_get_config_values()
        networks = self.validator_config.get_networks()
        self.nodes = {network: NodeFactory.create_node(network) for network in networks}
        self.block_height_cache = {network: self.nodes[network].get_current_block_height() for network in networks}
        self.challenge_factory = {
            NETWORK_BITCOIN: {
                MODEL_TYPE_BALANCE_TRACKING: BalanceChallengeFactory(self.nodes[NETWORK_BITCOIN])
            }
        }
        super(Validator, self).__init__(config)
        self.sync_validator()
        self.uid_batch_generator = get_uids_batch(self, self.config.neuron.sample_size)
        if config.enable_api:
            self.api_server = APIServer(
                config=self.config,
                wallet=self.wallet,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
                scores=self.scores
            )
        immunity_period = self.subtensor.immunity_period(self.config.netuid)
        bt.logging.info("Immunity period", immunity_period=immunity_period)
        self.miner_uptime_manager.immunity_period = immunity_period

    def resync_metagraph(self):
        super(Validator, self).resync_metagraph()
        self.sync_validator()

def run_api_server(api_server):
    api_server.start()


if __name__ == "__main__":
    from dotenv import load_dotenv

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    load_dotenv()

    with Validator() as validator:
        if validator.config.enable_api:
            if not validator.api_server:
                validator.api_server = APIServer(
                    config=validator.config,
                    wallet=validator.wallet,
                    subtensor=validator.subtensor,
                    metagraph=validator.metagraph,
                    scores=validator.scores
                )
            api_server_thread = threading.Thread(target=run_api_server, args=(validator.api_server,))
            api_server_thread.start()

        while True:
            logger.info("Validator API running")
            time.sleep(bt.__blocktime__ * 10)


