import argparse
from unstructured.partition.pdf import partition_pdf
from collections import Counter

def process_pdf(pdf_file):
    elements = partition_pdf(pdf_file)

    print(Counter(type(e) for e in elements))

    # concatenate text elements into a single string and save it to a txt file
    text = ''.join(e.text for e in elements if e.__class__.__name__ != 'Footer')
    with open(f'{pdf_file.split(".")[0]}.txt', 'w') as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file", help="input pdf file")
    args = parser.parse_args()

    process_pdf(args.pdf_file)
