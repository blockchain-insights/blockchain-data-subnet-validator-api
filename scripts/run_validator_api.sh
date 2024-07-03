#!/bin/bash

# Check if the api_key.json file exists and copy it with overwrite if it does
if [ -f /app/config/api_key.json ]; then
  cp -f /app/config/api_key.json /app/api_key.json
fi

# Check if the rate_limit.json file exists and copy it with overwrite if it does
if [ -f /app/config/rate_limit.json ]; then
  cp -f /app/config/rate_limit.json /app/rate_limit.json
fi

cd "$(dirname "$0")/../"

export PYTHONPATH=$PWD
python3 main.py --wallet.name "$WALLET_NAME" --wallet.hotkey "$WALLET_HOTKEY" --netuid "$NETUID" --subtensor.network "$SUBTENSOR_NETWORK" --subtensor.chain_endpoint "$SUBTENSOR_URL" --logging.trace
