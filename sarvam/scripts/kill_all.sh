PIDS=$(pgrep -f '/data/ASR/mayur/NeMo-asr/examples/asr/speech_multitask/speech_to_text_aed.py')
if [ -n "$PIDS" ]; then
    echo "Killing processes: $PIDS"
    kill $PIDS
    echo "Processes killed successfully."
else
    echo "No matching processes found."
fi

