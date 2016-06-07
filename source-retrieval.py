#!/usr/bin/env python3
""" Plagiarism detection source retrieval module.

    The code structure is based on the structure provided by
    Martin Potthast as a baseline for the PAN 2014 competition.
"""
__author__ = 'Rhulani Maluleka'
__version__ = '1.2'

import sys
import glob
import os
import codecs
import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, wordnet as wn
import string
from nltk import pos_tag
import simplejson
import unicodedata
import urllib.request
from bs4 import BeautifulSoup
import shingling
import time


# Constants
# =========

stop_words = set(stopwords.words("english")) | set(string.punctuation)
CHATNOIR = 'http://webis15.medien.uni-weimar.de/proxy/chatnoir/batchquery.json'
CLUEWEB = 'http://webis15.medien.uni-weimar.de/proxy/clueweb/id/'
SNIPPETURL = 'http://webis15.medien.uni-weimar.de/chatnoir/snippet'

PARAGRAPH_LEN = 5
QUERY_LEN = 10
TOP_QUERIES = 3
NUM_SHINGLES = 5
SIMILARITY_THRESHOLD = 5
TOP_NERs = 10

# Helper functions
# ================

def get_top_NEs(tagged_sentences, n=TOP_NERs):
    """ Return the n longest named entities of a text """
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))

    return sorted(entity_names, key=len, reverse=True)[:n]

def extract_entity_names(t):
    """ Extract the named entities from a given text """
    entity_names = []
    
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
                
    return entity_names

def strip_html(text):
    """ Strip html tags from a text """
    return BeautifulSoup(text, 'lxml').get_text()


class SourceRet:
    def process(self, suspdoc, outdir, token):
        """ Run the source retrieval pipeline. """
        # Extract the ID and initiate the log writer.
        self.init(suspdoc, outdir)
        # Read and tokenize the suspicious document.
        text = self.read_file(suspdoc)
        text_shingles = self.make_shingles(text)
        paragraphs = self.split_into_paragraphs(text)
        downloaded = []
          
        for i in range(len(paragraphs)):
            paragraphs[i] = self.preprocess(paragraphs[i])

        NEs = get_top_NEs(paragraphs)

        self.log_list(NEs)
        ne_results = self.submit_batch_queries(NEs, token)
        combined_ne_results = self.combine_results(ne_results)
        ranked_ne_results = self.rank_results(combined_ne_results,
                                               text_shingles)
        for result in ranked_ne_results:
            if (result[0] > SIMILARITY_THRESHOLD and
                result[1]['url'] not in downloaded):
                source = self.download_result(result[1], token)
                downloaded.append(source[1])
                self.log(source[1])
                if source[0]["oracle"] == "source":
                    #print(source[1])
                    break

        for p in paragraphs:
            queries = self.extract_queries(p)
            self.log_list(queries)
            
            results = self.submit_batch_queries(queries, token)
            combined_results = self.combine_results(results)
            ranked_results = self.rank_results(combined_results,
                                               text_shingles)

            for result in ranked_results:
                if (result[0] > SIMILARITY_THRESHOLD and
                    result[1]['url'] not in downloaded):
                    source = self.download_result(result[1], token)
                    downloaded.append(source[1])
                    self.log(source[1])
                    if source[0]["oracle"] == "source":
                        #print(source[1])
                        break

        # Close the log writer.
        self.teardown()


    def init(self, suspdoc, outdir):
        """ Sets up the output file in which the log events will be written. """
        logdoc = ''.join([suspdoc[:-4], '.log'])
        logdoc = ''.join([outdir, os.sep, logdoc[-26:]])
        self.logwriter = open(logdoc, "w")
        self.suspdoc_id = int(suspdoc[-7:-4])


    def teardown(self):
        self.logwriter.close()


    def read_file(self, suspdoc):
        """ Reads the file suspdoc and returns its text content. """
        f = codecs.open(suspdoc, 'r', 'utf-8')
        text = f.read()
        f.close()
        return text


    def split_into_paragraphs(self, text, n=PARAGRAPH_LEN):
        """ Splits text (single string) into a list of 5-sentence paragraphs """
        sentences = sent_tokenize(text)
        paragraphs = [' '.join(sentences[i:i+n])
                      for i in range(0, len(sentences), n)]
        return paragraphs


    def preprocess(self, text, tag=True):
        """ Takes a string and returns a tokenized list, sans stop words """
        tokenizer = RegexpTokenizer("[a-zA-Z'`]+")
        words = tokenizer.tokenize(text)

        for sw in stop_words.intersection(words):
            while sw in words:
                words.remove(sw)
        if tag:
            words = pos_tag(words)        
        return words


    def extract_queries(self, tagged_text, k= QUERY_LEN, n= TOP_QUERIES):
        """ Takes a list of keywords and returns a list of (up to)
        n, k-length queries.
        """
        
        # Keep only verbs, nouns and adjectives
        keep = ['RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'NN', 'NNP', 'NNPS', 'NNS', 'PRP', 'JJ', 'JJR', 'JJS']
        query = []
        seen = set()
        uniq_keys = [i[0] for i in tagged_text if i[0] not in seen
                     and i[1] in keep and not seen.add(i[0])]
        queries = [' '.join(uniq_keys[i:i+k]) for i in range(0, len(uniq_keys), k)]
        return queries[:n]


    def submit_batch_queries(self, queries, token):
        """ Pose the query to the ChatNoir search engine. """
        json_query = u"""
        {{
           "max-results": 3,
           "suspicious-docid": {suspdoc_id},
           "queries": [
             {{
               "query-string": "{query0}"
             }}""".format(suspdoc_id = self.suspdoc_id, query0 = queries[0])
        for q in queries[1:]:
            json_query += u""",
             {{
               "query-string": "{query}"
             }}""".format(query = q)
        json_query += u"""
           ]
        }
        """
        
        json_query = \
            unicodedata.normalize("NFKD", json_query).encode("ascii", "ignore")

        request = urllib.request.Request(CHATNOIR, json_query)
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", token)
        request.get_method = lambda: 'POST'
        try:
            response = urllib.request.urlopen(request)
            results = simplejson.loads(response.read())
            response.close()
            return results
        except urllib.request.HTTPError as e:
            error_message = e.read()
            sys.stderr.buffer.write(str(error_message))
            sys.exit(1)


    def combine_results(self, results):
        """ Combine retrieved json results into a list of (result, query)
        tuples """
        res = []
        for batch in results['chatnoir-batch-results']:
            for result in batch.get('result-data'):
                res.append((result, batch.get('chatnoir-query')))
        return res


    def rank_results(self, results, text_shingles):
        """  Sort results by the similarity of their snippets
        compared to the suspicious document """
        sort_list = []
        
        for result in results:
            snippet = self.get_snippet(result).get('snippet')
            if snippet:
                snippet = strip_html(snippet)
                snip_shingles = self.make_shingles(snippet, 5)
                similarity = self.get_similarity(snip_shingles, text_shingles)
                sort_list.append((similarity,)+ result)
            else:
                sort_list.append((0,) + result)

        return sorted(sort_list,
                      key=lambda sort_list:sort_list[0])

    def get_snippet(self, result_tuple):
        """ Returns a snippet of a document based of the query
        	that returned the particular result """
        id = result_tuple[0].get('longid')
        query = result_tuple[1].get('query-string')
        request_url=SNIPPETURL+'?id='+str(id)+'&query='+query

        try:
            response = urllib.request.urlopen(request_url)
            snippet = simplejson.loads(response.read())
            response.close()
            return snippet
        except urllib.request.HTTPError as e:
            error_message = e.read()
            #sys.stderr.write('Failed to get sippent for result with id:'+
            #                 str(id)+'\n')
            return self.get_empty_snippet(result_tuple)
            #sys.exit(1)

    def make_shingles(self, text, w=NUM_SHINGLES):
        """ Returns the w-shingles for a given text """
        text = " ".join(self.preprocess(text, False))
        sh = shingling.getInstance()
        shilgle_list = sh.wshingling(text, window_size=w)
        return {tuple(i) for i in shilgle_list}


    def get_similarity(self, snippet_shingles, text_shingles):
        """ Caculates the similarity of two texts based on their
            w-shingles """
        return len(snippet_shingles.intersection(text_shingles))


    def get_empty_snippet(self, result_tuple):
        """ Create dummy snippet for results that do not return
        one """
        json_query = u"""
        {{
           "snippet": "",
           "highlight": false,
           "query": "{query}",
           "length": 500,
           "id": {id}
        }}
        """.format(query = result_tuple[1]['query-string'],
                   id=result_tuple[0]['longid'])
        json_query = simplejson.loads(json_query)
        return json_query


    def download_result(self, result, token):
        """ Download a given result. """
        document_id = result["longid"]
        document_url = result["url"]
        request = urllib.request.Request(CLUEWEB + str(document_id))
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", token)
        request.add_header("suspicious-docid", str(self.suspdoc_id))
        request.get_method = lambda: 'GET'
        try:
           response = urllib.request.urlopen(request)
           download = simplejson.loads(response.read())
           response.close()
           return download, document_url
        except urllib.request.HTTPError as e:
           error_message = e.read()
           sys.stderr.write(error_message)
           sys.exit(1)

    def check_oracle(self, download):
        """ Checks is a given download is a true positive source document,
            based on the oracle's decision. """
        if download["oracle"] == "source":
            print("Success: a source has been retrieved.")
        else:
            print("Failure: no source has been retrieved.")


    def log(self, message):
        """ Writes the message to the log writer, prepending a timestamp. """
        timestamp = int(time.time())  # Unix timestamp
        self.logwriter.write(' '.join([str(timestamp), message]))
        self.logwriter.write('\n')


    def log_list(self, message_list):
        """ Writes messages to the log writer, prepending a timestamp. """
        for message in message_list:
            self.log(message)


# Main
# ====

if __name__ == "__main__":
    """ Process the commandline arguments. We expect three arguments: 
        - The path to the directory where suspicious documents are located.
        - The path to the directory to which output shall be written.
        - The access token to the PAN search API.
    """
    if len(sys.argv) == 4:
        suspdir = sys.argv[1]
        outdir  = sys.argv[2]
        token   = sys.argv[3]
        
        suspdocs = glob.glob(suspdir + os.sep + 'suspicious-document???.txt')
        for suspdoc in suspdocs:
            print("Processing\n",suspdoc)
            sr = SourceRet()
            sr.process(suspdoc, outdir, token)
    else:
        print('\n'.join(["Unexpected number of command line arguments.",
        "Usage: ./source_retrieval.py {susp-dir} {out-dir} {token}"]))
