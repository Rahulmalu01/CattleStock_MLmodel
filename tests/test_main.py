from main import build_parser


def test_train_command_is_available():
    args = build_parser().parse_args(["train"])

    assert args.command == "train"


def test_api_command_defaults():
    args = build_parser().parse_args(["api"])

    assert args.command == "api"
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.reload is False


def test_api_command_accepts_options():
    args = build_parser().parse_args(
        ["api", "--host", "127.0.0.1", "--port", "8080", "--reload"]
    )

    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.reload is True
