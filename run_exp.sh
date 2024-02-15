# Step 0. Change this to your university ID
UNIID='2000934580'
mkdir -p $UNIID

# Step 1. (Optional) Any preprocessing step, e.g., downloading pre-trained word embeddings


# Step 2. Train models on on CF-IMDB.
PREF='cfimdb'
python main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_output "${UNIID}/${PREF}-dev-output.txt" \
    --test_output "${UNIID}/${PREF}-test-output.txt" \
    --model "${UNIID}/${PREF}-model.pt"


# Step 3. Prepare submission:
##  3.1. Copy your code to the $UNIID folder
#for file in 'main.py' 'model.py' 'vocab.py' 'setup.py'; do
for file in 'main.py' 'model.py' 'vocab.py' ; do
	cp $file ${UNIID}/
done
##  3.2. Compress the $UNIID folder to $UNIID.zip (containing only .py/.txt/.pdf/.sh files)
##  3.3. Submit the zip file to Canvas! Congrats!