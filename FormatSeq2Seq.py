import json
import argparse
from os.path import join
from spacy.lang.en import English

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("dataset_file")
    parser.add_argument("split")
    parser.add_argument("output_dir")
    parser.add_argument("--spacy", default=True)

    args = parser.parse_args()

    with open(args.dataset_file) as inh:
        samples = json.load(inh)

    if args.spacy:
        nlp = English()

    with open(join(args.output_dir,'{}-src.txt').format(\
                                        args.split), 'w') as srch:
        with open(join(args.output_dir,'{}-tgt.txt').format(\
                                        args.split), 'w') as tgth:
            for sample in samples:
                src = ' ||| '.join(sample['History']+[sample['Question']])
                tgt = sample['Rewrite']
                if args.spacy:
                    src = ' '.join([tok.text for tok in nlp(src)])
                    tgt = ' '.join([tok.text for tok in nlp(tgt)])

                srch.write(src+'\n')
                tgth.write(tgt+'\n')

if __name__ == "__main__":
    main()

