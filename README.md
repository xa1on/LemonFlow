# LemonFlow

live speech to text recording and transcription via [lemonade-sdk](https://github.com/lemonade-sdk/lemonade) and [whisper.cpp](https://github.com/ggml-org/whisper.cpp)

## Features

- inference through local automatic speech recognition models (i just call it speech to text)
- live transcriptions w/ [whisper](https://github.com/openai/whisper) models
    - not just one massive blob of text after 30 seconds of talking, automatically transcribes speech as you pause and talk

## TODO

- [ ] string together results to create one unified script
- [ ] fancy local mini llm formatting (maybe in markdown or other formats)
- [ ] autocorrect punctuation and grammer
- [ ] dictionary to reference alternate words for specific pronounciations
- [ ] actually type out the stuff being transcribed via keyboard library
    - [ ] probably include a keybind for hold to activate
- [ ] turn into actual python package and distribute via pip?
    - [ ] alternatively, use something like pyinstaller to create actual application
    - [ ] possibly convert project to C++?