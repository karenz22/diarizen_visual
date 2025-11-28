from diarizen.pipelines.inference import DiariZenPipeline

# load pre-trained model
print("Loading pre-trained DiariZen model...")
diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")

# apply diarization pipeline
print("Applying diarization pipeline...")
diar_results = diar_pipeline('./example/EN2002a_30s.wav')

# print results
print("Diarization results:")
for turn, _, speaker in diar_results.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.0s stop=2.7s speaker_0
# start=0.8s stop=13.6s speaker_3
# start=5.8s stop=6.4s speaker_0
# ...

# load pre-trained model and save RTTM result
print("Loading pre-trained DiariZen model with RTTM output...")
diar_pipeline = DiariZenPipeline.from_pretrained(
        "BUT-FIT/diarizen-wavlm-large-s80-md",
        rttm_out_dir='.'
)
# apply diarization pipeline
print("Applying diarization pipeline with RTTM output...")
diar_results = diar_pipeline('./example/EN2002a_30s.wav', sess_name='EN2002a')
print(diar_results)