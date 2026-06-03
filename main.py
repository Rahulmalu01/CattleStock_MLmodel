#!/usr/bin/env python3
"""
CattleCare ML Service Entry Point
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="CattleCare ML Service")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    subparsers.add_parser("train", help="Train ML models")

    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="API host")
    api_parser.add_argument("--port", type=int, default=9000, help="API port")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.command == "train":
        print("Starting training pipeline...")
        from src.training import train_pipeline
        train_pipeline()

    elif args.command == "api":
        print(f"Starting API server on {args.host}:{args.port}")
        import uvicorn
        from src.api import app
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
