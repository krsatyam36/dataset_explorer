import pytest
from pathlib import Path
from datasets_explorer.storage import Storage
from datasets_explorer.tools import ToolExecutor
from datasets_explorer.utils import RateLimiter


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def storage(tmp_db_path: Path) -> Storage:
    return Storage(tmp_db_path)


@pytest.fixture
def tool_executor(storage: Storage) -> ToolExecutor:
    return ToolExecutor(
        storage=storage,
        query_id=1,
        rate_limiter=RateLimiter(),
        seen_urls=set(),
        subject="satellite imagery of military aircraft",
    )


@pytest.fixture
def tool_executor_generic(storage: Storage) -> ToolExecutor:
    return ToolExecutor(
        storage=storage,
        query_id=2,
        rate_limiter=RateLimiter(),
        seen_urls=set(),
        subject="cats and dogs",
    )
