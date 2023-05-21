from flask import Flask

import torch
import json
from transformers import BertForQuestionAnswering, BertTokenizerFast, pipeline, BertTokenizer

class SlotExtractor():
    def __init__(self):
        cuda = torch.cuda.is_available()
        if cuda:
            self.device = 0
        else:
            self.device = 'cpu'

        # init model
        self.tokenizer = BertTokenizer.from_pretrained('tokenizer')
        self.qa_model = BertForQuestionAnswering.from_pretrained('taxi_model')
        self.squad_pipeline = pipeline('question-answering', model=self.qa_model, tokenizer=self.tokenizer, device=self.device)
        with open('questions.json') as f:
            self.slots_to_questions = json.load(f)
    
        self.slots = ['taxi-arriveby', 'taxi-departure', 'taxi-destination', 'taxi-leaveat']

    def extract_slots(self, text):
        questions = [self.slots_to_questions[slot] for slot in self.slots]
        texts = [text] * len(questions)
        ans = self.squad_pipeline({'context': texts, 'question':questions})
        predicted_slots = {}
        for ans_, slot_ in zip(ans, self.slots):
            if ans_['score'] > 0.99:
                ans_text = ans_['answer'].strip()
                if ans_text in predicted_slots:
                    if predicted_slots[ans_text][1] < ans_['score']:
                        predicted_slots[ans_text] = (slot_, ans_['score'])
                else:
                    predicted_slots[ans_text] = (slot_, ans_['score'])

        return {slot_type[0]:slot_val for slot_val, slot_type in predicted_slots.items()}


