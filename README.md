# Wyoming SecureWakeWord
### Description
Home Assistant service that integrates Wyoming Protocol to the SecureWakeWord.
- Demo Video on **[Youtube](https://www.youtube.com/watch?v=F3AXUbL-i-o)**
- Research Paper on **[arXiv](https://arxiv.org/abs/2501.12194)**

### Details
- SecureWakeword is a voice authenticable wakeword system based on [OpenWakeWord](https://github.com/dscripka/openWakeWord).
- [Wyoming Protocol](https://github.com/rhasspy/wyoming) is a peer-to-peer protocol for voice assistants.

## Prerequisites
You can train and evaluate both wakeword and voiceauth models on [this repository](https://github.com/gws8820/securewakeword-model).

## Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/gws8820/wyoming-securewakeword.git
cd wyoming-securewakeword
script/setup
```

## Run

### 1. Place Your Custom Models
Ensure that your custom models are placed in the correct directories:

- **Wakeword model files**: `./wyoming-securewakeword/wyoming_securewakeword/custom_models/`
- **Voice authentication files**: `./wyoming-securewakeword/wyoming_securewakeword/voiceauth/`

### 2. Start the Server
Run the server to accept connections:
``` sh
script/run \
        --uri 'tcp://0.0.0.0:10400' \
        --preload-model 'YOUR_WAKEWORD_NAME' \
        --wake-threshold 0.5 \
        --auth-threshold 0.5
```

See `script/run --help` for more options.

## License
This project uses [wyoming-openwakeword](https://github.com/rhasspy/wyoming-openwakeword) developed by [rhasspy](https://github.com/rhasspy).

See [LICENSE](LICENSE) for details.
