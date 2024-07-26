from typing import Any, List, Dict
from sqlalchemy import Column, Integer, String, DateTime, Float, inspect, MetaData, text, UniqueConstraint, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.future import select
from datetime import datetime
import hashlib
from sqlalchemy.exc import OperationalError
from loguru import logger
from pydantic import BaseModel, Field
from .dbscan import dbscan

Base = declarative_base()

class Receipts(Base):
    __tablename__ = 'receipts'
    receiptid = Column(Integer, primary_key=True)
    validator_hotkey = Column(String, nullable=False)
    miner_hotkey = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    execution_time = Column(Float)
    prompt_hash = Column(String, nullable=False)
    prompt_preview = Column(String, nullable=False)
    completion_tokens = Column(Integer, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

class Miners(Base):
    __tablename__ = 'miner_blacklist'
    miner_hotkey = Column(String, primary_key=True)
    uid = Column(Integer, nullable=False)
    reason = Column(String)

class PromptHistoryRequest(BaseModel):
    miner_hotkeys: List[str] = Field(default=[], title="Miner hotkeys")
    query_start_times: List[str] = Field(default=[], title="Query started time")
    execution_times: List[float] = Field(default=[], title="Query execution time")
    prompt: str = Field(default="", title="executed prompt")
    token_usages: List[Dict[str, Any]] = Field(default=[], title="Token count")

class ReceiptManager:
    def __init__(self, db_url='sqlite:///./test.db'):
        self.engine = create_engine(db_url, echo=True)
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)
        self.verify_database()

    def verify_database(self):
        try:
            with self.engine.connect() as conn:
                inspector = inspect(conn)
                tables = inspector.get_table_names()
                if 'receipts' not in tables or 'miner_blacklist' not in tables:
                    raise OperationalError("Required tables are missing in the database.")
        except OperationalError as e:
            logger.error(f"Database verification failed: {e}")
            raise

    def check_miner_blacklisted(self, miner_hotkey: str):
        with self.Session() as session:
            try:
                query = select(Miners).where(Miners.miner_hotkey == miner_hotkey)
                result = session.execute(query)
                miner = result.scalars().first()
                if miner:
                    return True, miner.reason
                return False, 'Success'
            except Exception as e:
                logger.error(f'Error occurred while checking miner blacklist: {{"exception_type": {e.__class__.__name__}, "exception_message": {str(e)}, "exception_args": {e.args}}}')
                return True, 'Exception'

    def add_prompt(self, validator_hotkey: str, miner_hotkey: str, prompt: str, timestamp: datetime, execution_time: float, token_usage: dict) -> bool:
        with self.Session() as session:
            try:
                query = select(Receipts).where(
                    Receipts.validator_hotkey == validator_hotkey,
                    Receipts.timestamp == timestamp
                )
                logger.info(f'Query {query}')
                result = session.execute(query)
                existing_prompt = result.scalars().first()
                if existing_prompt:
                    logger.info('Prompt history already exists')
                    return False

                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                new_receipt = Receipts(
                    validator_hotkey=validator_hotkey, miner_hotkey=miner_hotkey,
                    prompt_preview=prompt[:64], prompt_hash=prompt_hash,
                    timestamp=timestamp, execution_time=execution_time,
                    completion_tokens=token_usage['completion_tokens'],
                    prompt_tokens=token_usage['prompt_tokens'],
                    total_tokens=token_usage['total_tokens']
                )
                session.add(new_receipt)
                session.commit()
                logger.info('Add prompt history success')
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Error occurred while recording prompt: {e.__class__.__name__}, Message: {str(e)}")
                return False

    def add_prompt_history(self, validator_hotkey: str, prompt_entry: PromptHistoryRequest):
        for miner_hotkey in prompt_entry.miner_hotkeys:
            blacklisted, msg = self.check_miner_blacklisted(miner_hotkey)
            if blacklisted:
                logger.info(f'Miner {miner_hotkey} is blacklisted: {msg}')
                return

        valid_miners = dbscan(prompt_entry)
        logger.info(f'Valid miners: {valid_miners}')
        for miner in valid_miners:
            self.add_prompt(
                validator_hotkey,
                miner[0],
                prompt_entry.prompt,
                datetime.fromisoformat(miner[1]),
                miner[2],
                miner[3]
            )