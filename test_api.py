import tensorflow as tf
import pickle
from model import Model
from utils import build_dict, build_dataset, batch_iter, build_datapoint


with open("args.pickle", "rb") as f:
    args = pickle.load(f)

print("Loading dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid", args.toy)
# print("Loading validation dataset...")

print("*************")
print("inp_seq_size: " + str(article_max_len))
print("*************")
print("out_seq_size: " + str(summary_max_len))
print("*************")
sess = tf.Session()
print("Loading saved model...")
model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state("./saved_model/")
saver.restore(sess, ckpt.model_checkpoint_path)

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


#!flask/bin/python
from flask import Flask
from flask import request
from flask import abort
from flask import jsonify


app = Flask(__name__)

@app.route('/summarization/abstractive/', methods=['POST'])
def abs_lee():
    # print(request.data)
    text = request.data.decode('utf-8', 'ignore')
    print("_________INPUT__________")
    print(text)
    print("________________________")
    clean_sents = []
    sents = sent_tokenize(str(text))
    for i in range(len(sents)):
        words = word_tokenize(sents[i])
        clean_sents.append(" ".join(words))
    clean_text = (" ".join(clean_sents)).lower()
    # print("_________CLEAN INPUT__________")
    # print(clean_text)
    # print("________________________")
    valid_x = build_datapoint("valid", word_dict, article_max_len, summary_max_len, clean_text, args.toy)
    valid_x_len = [len([y for y in x if y != 0]) for x in valid_x]

    batches = batch_iter(valid_x, [0] * len(valid_x), 1, 1)

    # print("Writing summaries to 'result.txt'...")
    for batch_x, _ in batches:
        batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]

        valid_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
        }

        prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
        prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]

        for line in prediction_output:
            summary = list()
            for word in line:
                if word == "</s>":
                    break
                if word not in summary:
                    summary.append(word)
            s = " ".join(summary)


    out = {"summary":str(s)}
    return jsonify(out)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False,port=8997)
