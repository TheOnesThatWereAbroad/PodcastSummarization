import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main(args):

    print("Reading the transcript...\n")
    with open(args.transcript, 'r', encoding="utf8") as f:
        transcript = f.read()
    
    print("Generating summary...\n")
    tokenizer = AutoTokenizer.from_pretrained("gmurro/bart-large-finetuned-filtered-spotify-podcast-summ")
    model = AutoModelForSeq2SeqLM.from_pretrained("gmurro/bart-large-finetuned-filtered-spotify-podcast-summ", from_tf=True)

    input_ids = tokenizer(transcript, truncation=True, return_tensors="pt").input_ids

    gen_kwargs = {
        "length_penalty": 2.0,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "min_length": 39,
        "max_length": 250
    }
    outputs = model.generate(input_ids, **gen_kwargs)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nSUMMARY:\n{summary}")

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("transcript", help="Path of the file containing the transcript", action="store")

    # Optional argument flag which defaults to False
    parser.add_argument("-f", "--filter", help="Whether apply transcript filtering before summarization", action="store_true", default=False)


    args = parser.parse_args()
    main(args)