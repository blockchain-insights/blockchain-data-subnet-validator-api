from fastapi import Request, HTTPException
from starlette.responses import JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
import time
from loguru import logger


class RateLimiter:
    def __init__(self):
        self.clients = {}

    def is_rate_limited(self, client_ip: str, max_requests: int) -> bool:
        current_time = time.time()
        if client_ip not in self.clients:
            self.clients[client_ip] = []
            logger.info(f"New client added: {client_ip}")

        # Filter out timestamps outside the window
        self.clients[client_ip] = [timestamp for timestamp in self.clients[client_ip] if current_time - timestamp < 60]

        if len(self.clients[client_ip]) >= max_requests:
            logger.warning(f"Rate limit exceeded for client: {client_ip}")
            return True

        self.clients[client_ip].append(current_time)
        logger.info(f"Request recorded for client: {client_ip} at {current_time}")
        return False


rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next, max_requests: int):
    client_ip = request.client.host
    if rate_limiter.is_rate_limited(client_ip, max_requests):
        logger.error(f"Too Many Requests from client: {client_ip}")
        return JSONResponse(status_code=HTTP_429_TOO_MANY_REQUESTS, content={"detail": "Too Many Requests"})

    response = await call_next(request)
    return response
