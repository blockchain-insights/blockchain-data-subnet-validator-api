from substrateinterface import Keypair

def sign_message(keypair: Keypair, request: dict):
    message = '.'.join(str(getattr(request, attr)) for attr in dir(request) if
                    not callable(getattr(request, attr)) and not attr.startswith("__"))

    signature = f"0x{keypair.sign(message).hex()}"
    
    return signature