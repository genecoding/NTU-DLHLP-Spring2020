for split in train dev test;
do
  python3 scripts/convert_conll_to_raw.py /content/ctb/$split.conllx > /content/ctb/$split.txt
  python3 scripts/convert_raw_to_bert.py /content/ctb/$split.txt /content/ctb/$split.bertbase-layers.hdf5 chinese
done
