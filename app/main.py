import json
import math
from collections import Counter
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from.data_models.summary_report_model import SummaryReport, TranscriptSummary
from .data_models.conversation_model import Transcript
from .data_models.emotions_report_model import EmotionsReport, SpeakerEmotions
from .data_models.speaker_relationship_report import SpeakerRelationship, SpeakerRelationshipsReport
from .ml_models.summary_model.model import SummaryModel
from .ml_models.emotion_model.model import EmotionModel
from .ml_models.encoder_model.model import TextEncoderModel 
from .ml_models.clustering_model.model import TextClusteringModel
from .ml_models.agreement_model.model import TextAgreementModel
from .config import DECAY_FACTOR

load_dotenv()

summary_model = SummaryModel()
emotion_model = EmotionModel()
text_encoder_model = TextEncoderModel()
text_clustering_model = TextClusteringModel()
text_agremment_model = TextAgreementModel()

app = FastAPI()

@app.get('/')
async def health_check():
    return {'status': 'OK'}

@app.post('/summarize')
async def summarize_transcript(transcript : Transcript) -> SummaryReport:
    summary = summary_model.predict_summmary(transcript=transcript)
    summary = json.loads(summary)
    summary = TranscriptSummary(**summary)
    summary_report = SummaryReport(report=summary)
    return summary_report

@app.post('/emotions')
async def emotions_analysis(transcript : Transcript) -> EmotionsReport:
    conversation = transcript.conversation
    speakers = set([dialog.speaker for dialog in conversation])
    dialogs = [i for i in conversation if len(i.message.split()) >= 3]
    emotions = emotion_model.predict_emotions(dialogs)
    speaker_to_emotion_mapper = {speaker: Counter() for speaker in speakers}
    for emotion, dialog in zip(emotions, dialogs):
        speaker_to_emotion_mapper[dialog.speaker][emotion] += 1
    data = []
    for speaker, emotions_counter in speaker_to_emotion_mapper.items():
        speaker_top_emotions = emotions_counter.most_common(2)
        while len(speaker_top_emotions) < 2:
            speaker_top_emotions.append((None, None))
        primary_emotion, secondary_emotion = speaker_top_emotions
        data.append(SpeakerEmotions(speaker=speaker, 
                                    primary_emotion=primary_emotion[0],
                                    secondary_emotion=secondary_emotion[0]))
    emotions_report = EmotionsReport(report=data)
    return emotions_report

@app.post('/relationship')
async def analyse_speaker_relationships(transcript : Transcript, speaker : str = Query(None)) -> SpeakerRelationshipsReport:
    conversation = transcript.conversation
    speakers = set([dialog.speaker for dialog in conversation])
    if speaker not in speakers:
        return HTTPException(status_code=400, detail='Invalid speaker: Speaker not present in the conversation')
    dialogs = []
    for idx, dialog in enumerate(conversation):
        if len(dialog.message.split()) >= 3:
            dialog.__setattr__('index', idx)
            dialogs.append(dialog)
    texts = [dialog.message for dialog in dialogs]
    embeddings = text_encoder_model.calculate_embeddings(texts)
    cluster_labels = text_clustering_model.calculate_clusters(embeddings)
    clusters = {}
    for dialog, label in zip(dialogs, cluster_labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(dialog)
    dialog_pairs = []
    for dialogs in clusters.values():
        for i in range(0, len(dialogs)):
            for j in range(i+1, len(dialogs)):
                dialog1 = dialogs[i]
                dialog2 = dialogs[j]
                if (dialog1.speaker == speaker and dialog2.speaker != speaker) or (dialog2.speaker == speaker and dialog1.speaker != speaker):
                    dialog_pairs.append((dialog1, dialog2))
    speaker_agreement_scores = {i : [] for i in speakers if i != speaker}
    max_possible_agreement_scores = {i: [] for i in speakers if i != speaker}
    for pair in dialog_pairs:
        label, init_score = text_agremment_model.predict_category(pair[0].message, pair[1].message)
        if label == 1:
            init_score = -init_score
        distance = abs(pair[0].index - pair[1].index)
        weight = math.e**-(DECAY_FACTOR*(distance-1))
        agreement_score = init_score * weight
        max_possible_agreement_score = 1 * weight
        other_speaker = pair[0].speaker if pair[0].speaker != speaker else pair[1].speaker
        speaker_agreement_scores[other_speaker].append(agreement_score)
        max_possible_agreement_scores[other_speaker].append(max_possible_agreement_score)
    data = []
    for other_speaker in speakers: 
        if other_speaker != speaker:
            agreement_score = sum(speaker_agreement_scores[other_speaker])/sum(max_possible_agreement_scores[other_speaker]) if len(max_possible_agreement_scores[other_speaker]) > 0 else 0
            data.append(SpeakerRelationship(speaker1=speaker,
                                        speaker2=other_speaker,
                                        agreement_score=agreement_score))
        
    print(data)
    speaker_relationships_report = SpeakerRelationshipsReport(report=data)
    return speaker_relationships_report