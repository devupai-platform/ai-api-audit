#!/usr/bin/env bash
# Runs ai_api_audit.py against every mock mode and checks the expected verdicts.
set -u
cd "$(dirname "$0")"
python3 - <<'PY'
from tokenizers import Tokenizer, models, pre_tokenizers
t = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
t.pre_tokenizer = pre_tokenizers.Whitespace()
t.save("tok.json")
PY
export AI_API_KEY=test-key
declare -A EXPECT=(
  [honest]="0 FAIL"
  [substitute]="prompt counted with the model's tokenizer"
  [no_usage]="usage returned"
  [inflate_output]="completion tokens match delivered text"
  [hidden_prompt]="no large hidden text added"
  [truncate]="advertised context window served"
  [ignore_max]="max_tokens enforced"
  [wrong_model]="response names the requested model"
  [weak_recall]="advertised context window served"
)
port=18100; ok=0; bad=0
for mode in honest substitute no_usage inflate_output hidden_prompt truncate ignore_max wrong_model weak_recall; do
  port=$((port+1))
  python3 mock_provider.py "$mode" "$port" tok.json & pid=$!
  sleep 0.6
  out=$(python3 ../ai_api_audit.py --base-url "http://127.0.0.1:$port/v1" --model acme/real-model-70b \
        --tokenizer-file tok.json --context-test 6000 --yes 2>&1); code=$?
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  want="${EXPECT[$mode]}"
  if [ "$mode" = honest ]; then
    if [ $code -eq 0 ] && ! grep -q '^\[WARN\]' <<<"$out"; then r=OK; else r=BAD; fi
  else
    need=1; case "$mode" in hidden_prompt|weak_recall) need=0;; esac   # these are WARNs, not FAILs
    if [ $code -eq $need ] && grep -E "^\[(FAIL|WARN)\] $want" <<<"$out" >/dev/null; then r=OK; else r=BAD; fi
  fi
  [ $r = OK ] && ok=$((ok+1)) || bad=$((bad+1))
  echo "=== $mode → $r (exit $code)"; echo "$out" | sed -n '3,20p'
done
echo; echo "SUMMARY: $ok OK, $bad BAD"
