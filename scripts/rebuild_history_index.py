#!/usr/bin/env python3

from __future__ import annotations

import json

from services.history import get_history_store


def main() -> None:
    store = get_history_store()
    info = store.rebuild_index()
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

