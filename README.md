# Erroneous-Old-German-Text-Correction
    Based on deep learning, correct the erroneous old German Text with limited dataset.
##      Task:Apply seq2seq model to correct old German sentence with character or segmentation errors.
###     1. Generate gound-truth data semi-automatically.
###     2. Synthesize GT: give a clean text, try to mimic the error patterns(using AutoEncoder).
###     3. Apply existing models(e.g.transformer/bert) of encoder-decoder framework to generate candidate sentence with above 2types of GT.
