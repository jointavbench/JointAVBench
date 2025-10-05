task_instructions = {
    "task1":"Generate a question-answer pair that identifies the timestamp of a specific element (object or character). You can use the following question format: \"Which objects appear only in the dialogue and not in the video, and when does the first object appear in the dialogue?\". Note that terms such as characters, names and professions are not physical objects. If there's no objects mentioned in speech transcript, output '[Unavailable]' and stop.",
    "task2":"Generate a question-answer pair that identifies the spatial location of a movie character in the video. Make sure to use both the video caption and auditory content(speech emotion) in designing the question. The question should ask like: \"Where's the character that says <speech content> located in the video?\" If there's only one character in the video, output '[Unavailable]'. Please ensure that the speaker traits match the person in the video. If there are multiple video characters that meet the speaker traits or no matching character exists, output '[Unavailable]'.",
    "task3":"Generate a question-answer pair that identifies the spatial location of a specific object/animal/place mentioned or referred to in the subtitle. The answer must be based on BOTH visual information (video caption) AND speech information (subtitle). Remember to create the question-answer pair ONLY when the object/animal/place referred to in the speech transcript is clear and unambiguous.",
    "task4":"Generate a question-answer pair that identifies the spatial location of a sounding object in the video. The question should ask like: (1) \"What is the spatial position of the object that produced the <sound> in the scene?\" (2) \"Where's the sounding object located in the video?\" If the object's position cannot be reliably inferred from video captions and sound events, output '[Unavailable]' and stop.",
    "task5":"Generate a question-answer pair that infers what event occurred in both video frames and audio. The question should ask like: \"What makes the <sound characteristic> sound?\" If there're no obvious connections between sound and the object or the actions to make the sound isn't certain, output '[Unavailable]' and stop.",
    "task6":"Generate a question-answer pair that identifies the emotions of the characters based on the video and speech emotion. You can use the following question format: (1) \"What's the emotion of the speaker that <speaker appearance>?\" (2) \"What emotion is conveyed by the speaker that <speaker appearance>?\" Please ensure that the speaker traits in the speech emotion match the person in the video. If there are multiple video characters that meet the speaker traits or no matching character exists, output '[Unavailable]'. If the video character matches with the speaker traits, use the apperance of the video character to create questions.",
    "task7":"Generate a question-answer pair that identifies the overall tone of the scene based on the background music and video. You can use the following question format: \"What is the overall atmosphere of the scene?\" Output '[Unavailable]' and stop if the question can be easily answered using only the music or the video. Remember to use the music and video in determining the scene's atmosphere.",
    "task8":"Generate a question-answer pair that identifies the sequential order of elements based on information in the video and character dialogue. You can use the following question format: \"In what order were the following items mentioned in the video? (a)[element 1] (b)[element 2] (c)[element 3]\" If the video caption contains less than 4 transitions or the subtitle contains less than 4 sentences, output '[Unavailable]' and stop. Remember to use the element(object in the video caption or subtitle phrase) from both the video caption and the subtitle. Ensure the sequential order between these elements is clear and can be obtained from the input material. If the element sequence isn't clear and unambiguous, output '[Unavailable]'",
    
    "task9":"identifies the association of a particular element across different segments, such as an item mentioned by a character in a previous segment appearing in a later segment based on the provided video and audio segment captions. You can use the following question format: (1)\"Which dialogue in other segments is most relevant to what the <character traits> does in Segment <idx>?\"(2)\"Which conversation in other clips is most related to the <character traits>'s actions in segment <idx>?",
    "task10":"identifies the sequential order of different audio-visual plot information in different segments. The details to be sorted should include both audio and visual elements, each detail should include information from only one modality. You can use the following question format: \"In what order were the following items mentioned in the video? (a)[element 1] (b)[element 2] (c)[element 3]\" You should use elements from subtitles, speech emotion, and video captions across different segments as the 3 items to be sorted. Additionally, please ensure that the elements to be ranked are sourced separately from subtitles, speech emotion, and video captions",
    "task11":"identifies the time point of the segment based on the description of the question combined with audio and video information. You can use the following question format: (1) \"When did <plot of the segment> happen in the video?\" (2)\"At what point in the video does <plot of the segment> occur?\" Remember to output the specific segment id as answer. Ensure that the plot of the segment in the question uses information from subtitle.",
    "task12":"identifies the role of sound effects and background music across different video segments. You can use the following question format: (1) \"[Question] How does the video depict the <character action> of <character traits>? [Answer] By using the <sound characteristics> multiple times\"  (2) \"What's the function of the music?\" (3) \"What role does the shift in music play in the storyline?\" Ensure that the music or the sound event spans multiple segments. Note that the question should utilize information from both audio and video modality. Additionally, please ensure that the question does not contain any hint to the answer (eg. question revealing the exact action that caused the sound).",
    "task13":"identifies the result of a continuous event that spans across multiple segments. The question should ask like: \"How did the cross-segment continuous event conclude in segment <idx>?\" The event should include information from both video caption and subtitle. Ensure that the event spans as much segments as possible.",
    
    "task15":"tests the model's long-term memory ability, specifically whether the model can remember some detailed information after watching the entire video segments. You can use detailed information from any segment. Remember not to directly ask about the element but to utilize both the visual and the auditory information to create detailed and precise questions.",
    "task16":"answers the emotional changes and trajectory of the entire movie based on the background music and video of the scenes. Ensure that the constructed question-answer pairs cover the emotional changes across most segments of the video, rather than just a few consecutive segments. You should utilize both musical emotion and visual atmosphere to create questions.",
    "task17":"infers the relationship between characters in the movie based on the overall plot. Ensure that the inferred character is not a supporting role and appears in multiple segments, thus providing sufficient information for relationship inference.",
}

task_examples = {
    # "task1":"[Question] When did the character talked about the object that appeared in the video?\n[Answer] At around 00:03:12\n[Explanation] The object that the characters talked about is the shirt, which also appearred in the video caption. From the speech transcript, we can see that the shirt was mentioned at around 00:03:12. So the answer is at around 00:03:12.\n[Question] When was the person in white started talking about going out?\n[Answer] At around 00:03:12\n[Explanation] There's only one man in white in the video caption. The speech emotion suggests there's a man said 'I'd like to go out', and the same sentence appeared in speech transcript at 00:03:12. So the answer is 00:03:12.",
    "task1":"[Question] Which objects appear only in the dialogue and not in the video, and when does the first object appear in the dialogue?\n[Answer] The shirt and the hat, mentioned at around 23s\n[Explanation] The objects that the characters talked about and not appearred in the video is the shirt and the hat. So the first one is the shirt. From the speech transcript, we can see that the shirt was mentioned at around 23s. So the answer is at around 23s.",
    "task2":"[Question] Where's the character that says 'I'm all right' with a joyful tone located in the video?\n[Answer] In the left\n[Explanation] The video caption suggests there's a man and a woman in the video, the man is in the left and the woman is in the right. The speech emotion suggests that the speech content 'I'm all right' is delivered by a male, which aligns with the man in the left. So the speaker is located in the left.",
    "task3":"[Question] Where is the object mentioned in the subtitles located in the video?\n[Answer] In the man's hand\n[Explanation] The object that appeared both in video caption and speech transcription is the apple. The subtitle text mentioned 'Where's the apple?'. And from video caption, the man is holding the apple. So the answer is 'In the man's hand'.",
    "task4":"[Question] Where's the sounding object located in the video?\n[Answer] On the road.\n[Explanation] The sounding object in the sound caption is the vehicle. According to the video caption, the vehicle in the scene is located on the road. So the answer is on the road.",
    "task5":"[Question] What makes the medium pitch sound?\n[Answer] Flame\n[Explanation] The sound event caption described the fire flame with a medium pitched sound. And there's flame shown in the video caption, which can support the sound event caption. So the answer is flame.\n",
    "task6":"[Question] What's the emotion of the speaker that wear glass?\n[Answer] Happy\n[Explanation] In the video, there's only one man with glass. In the speech emotion, the man's voice is happy. So the person's emotion is happy.",
    "task7":"[Question] What is the overall atmosphere of the scene?\n[Answer] Pleasant\n[Explanation] The music has a mood of cheerful and pleasant and the video caption shows that the scene is colorful. So based on the details the emotion is pleasant.",
    # "task8":"[Question] In what order were the following items shown or mentioned in the video? (a) bacon (b) stick (c) cigarettes\n[Answer] (b) (c) (a)\n[Explanation] The video caption shows that a boy is holding a stick, then the camera view shifts and show that a pack of cigarettes. The subtitle in the end mentioned the bacon. So the correct answer is (b)(c)(a).",
    "task8":"[Question] In what order were the following items shown or mentioned in the video? (a) A boy holding a stick (b) 'Pass the cigarettes' (c) 'I need bacon!' \n[Answer] (a) (b) (c)\n[Explanation] The video begins with the boy holding a stick (a). In the end of the video, the subtitle 'Pass the cigarettes' (b) appears. The final subtitle mentions bacon (c).",
    
    "task9":"<Output 1>\n[Question] Which dialogue in other segments relates to what the boy does in Segment 7?\n[Answer] 'Let's play poker game!'\n[Explanation] The video caption showed that the boy is playing poker game in segment 7. In segment 5, the speech transcript shows 'Let's play poker game!', so the most reasonable answer is 'Let's play poker game!'",
    "task10":"<Output 1>\n[Question] In what order were the following items mentioned in the video? (a) The man is riding a bike (b) The man talked about cars with a joyful tone (c) 'Let's play basketball'.\n[Answer] (a) (c) (b)\n[Explanation] In segment 7, the main content is that a man is riding a bike. In segment 9, the subtitle contains the dialogue 'Let's play basketball'. And in segment 13, the speech content in speech emotion is about cars and the speech emotion is joyful. So the answer is (a)(c)(b).",
    "task11":"<Output 1>\n[Question] At what point in the video does the boy in a gray sweater held his school bag before talking about homework?\n[Answer] Segment 13\n[Explanation] The boy talked about his homework only in segment 17 from the speech transcript, before that the last time the boy holding a bag was in segment 15 from the video caption. So the answer is Segment 15.",
    "task12":"<Output 1>\n[Question] How does the video depict the movement of the man in orange sweater?\n[Answer] By using the footstep sound in multiple segments\n[Explanation] From the audio transcript, the sound of footstep appeared in segment 11, 13, 17, 19. In these segments, the man in orange sweater was walking. So the answer is By using the footstep sound in multiple segments\n<Output 2>\n[Question] What's the function of the music?\n[Answer] It is used to express the transformation of young boy's emotions\n[Explanation] The music in segment 13 was cheerful, and the young boy was smiling in the video. But the music in segment 21 was melancholic, where the young boy looked down. So the answer is 'It is used to express the transformation of young boy's emotions'.",
    "task13":"<Output 1>\n[Question] How did the cross-segment continuous event conclude in segment 8?\n[Answer] Get some medicine\n[Explanation] According to the speech transcript in segment 5, someone seems to be in a fever and the boy said to help the person. Then from the video caption in segment 7, the boy is on the road. And in segment 10, the boy get the medicine and come back. So the answer is get some medicine.",
    
    "task15":"<Output 1>\n[Question] Where was the boy when someone mentioned 'watermelon'?\n[Answer] Sitting on a table\n[Explanation] According to the subtitle, someone mentioned watermelon only in segment 25. From the video caption in segment 25, a boy was sitting on a table. So the answer is 'sitting on a table'.",
    "task16":"<Output 1>\n[Question] What's the emotion transformation of the last half of the movie?\n[Answer] From calm to sad\n[Explanation] The audio caption in the middle few segments showed that the mood is calm and peace, while the last few segments showed that the main mood become sad and miserable. So the answer is from calm to sad",
    "task17":"<Output 1>\n[Question] What's the relationship between the man with beard and glass and the woman in pink sweater? \n[Answer] The man and the woman are a couple.\n[Explanation] The video caption showed that the man and the woman are very close and often appear together. The subtitle showed that the man call the woman wife. Based on that, the man and the woman are most likely a couple.",
}
task_modalities = {
    "task1":"Video caption: {video_caption}\nSpeech transcript: {subtitle}\n",
    "task2":"Video caption: {video_caption}\nSpeech Emotion: {speech_emotion}\n",
    "task3":"Video caption: {video_caption}\nSpeech transcript: {subtitle}\n",
    "task4":"Video caption: {video_caption}\nSound Event: {sound_event}\n",
    "task5":"Video caption: {video_caption}\nSound Event: {sound_event}\n",
    "task6":"Video caption: {video_caption}\nSpeech Emotion: {speech_emotion}\n",
    "task7":"Video caption: {video_caption}\nMusic: {music}\n",
    "task8":"Video caption: {video_caption}\nSpeech transcript: {subtitle}\n",
    
    "task9":"Video caption: {video_caption}\nSpeech transcript: {subtitle}\nSpeech Emotion: {speech_emotion}\n",
    "task10":"Video caption: {video_caption}\nSpeech transcript: {subtitle}\nSpeech Emotion: {speech_emotion}\n",
    "task11":"Video caption: {video_caption}\nSpeech transcript: {subtitle}\nSpeech Emotion: {speech_emotion}\n",
    "task12":"Video caption: {video_caption}\nMusic: {music}\nSound Event: {sound_event}\n",
    "task13":"Video caption: {video_caption}\nSpeech transcript: {subtitle}\nSpeech Emotion: {speech_emotion}\n",
    
    "task15":"Story Description: {video_description}\nVideo caption: {video_caption}\nSpeech transcript: {subtitle}\nSpeech Emotion: {speech_emotion}\nMusic: {music}\nSound Event: {sound_event}\n",
    "task16":"Story Description: {video_description}\nVideo caption: {video_caption}\nMusic: {music}\n",
    "task17":"Story Description: {video_description}\nVideo caption: {video_caption}\nSpeech transcript: {subtitle}\nSpeech Emotion: {speech_emotion}\n",
}

single_scene_universal_prompt = """
Your task is to generate a question-answer pair based on the instructions. The question must utilize both visual and audio information (e.g. speech, sound event, or music). Generate a question-answer pair based on this analysis, ensuring the question does not contain any hints or details about the answer. The answer should be precise and concise, and include an explanation justifying the answer. If the material does not provide sufficient information to generate a valid question-answer pair, respond with '[Unavailable]'. 
Please follow the instructions below:\n{instruction}
Format the output as follows:
[Question]......
[Answer]......
[Explanation]......
Example:\n{examples}
Here are the input material:\n{{segments_info}}
Please follow the instructions and refer to the examples provided to assist in your question design.
""".strip()

multi_scene_universal_prompt = """
Your task is to generate four different question-answer pairs based on the instructions. Each question must integrate information from both the audio and video modalities, ensuring that neither modality alone can provide the answer. Importantly, the information used to formulate each question should be derived from a few consecutive segments of the material, rather than from a single segment or the entire content.
The generated question-answer pairs should be unique and avoid overlapping in content. The questions should be designed without revealing hints or details about the answers. The answers should be precise and concise, accompanied by a brief explanation that justifies the response based on the combined audio-visual information from the selected segments. Use the video description to enhance your understanding of the material.
Format the output as follows:
<Output id>
[Question]......
[Answer]......
[Explanation]......
Please follow the instructions below:
Generate four different question-answer pairs that {instruction}
Example:\n{examples}
Here is the input material:\n{{segments_info}}
Please follow these instructions and refer to the examples provided to guide your question design.
"""
full_scene_universal_prompt = """
Your task is to generate four different question-answer pairs based on the instructions. The questions must utilize both visual and audio information, and cannot be answered by information from only one modality. The generated four questions should be different from each other. The questions should also be derived based on the whole movie, not just a few segments. Generate the question-answer pairs based on this analysis, ensuring each question does not contain any hints or details about the answer. The answers should be precise and concise, and include an explanation justifying the answer for each question. You can use the video description to help you better understand the material. 
Format the output as follows:
<Output id>
[Question]......
[Answer]......
[Explanation]......
Please follow the instructions below:
Generate four different question-answer pairs that {instruction}
Example:\n{examples}
Here are the input material:
{{segments_info}}
Please follow the instructions and refer to the examples provided to assist in your question design.
""".strip()

# general check
qa_modality_judge_universal_prompt = """
You are a multimodal understanding assistant. You have access to the following:
1. Question: A question related to the video clip. 
2. Answer: An answer provided to the question.
3. Explanation: An explanation supporting the answer.
Your task includes the identification of 2 different modalities, video(video caption) and audio(subtitle text, speech emotion, sound event and music). 
Your task is to evaluate whether the question can be answered using only one modality (either video or audio) or if it requires both modalities. Please strictly base your judgment on the information explicitly required to answer the question, as well as the content of the provided answer and explanation. Avoid making assumptions about the content of the modalities beyond what is explicitly stated in the question, answer, or explanation.
Please follow these steps to complete your evaluation:
1. Information Analysis: 
    Analyze the question to identify the specific visual and auditory details required to answer it. Note that explicitly indicating the exclusion of specific visual or auditory information implies a need for such information (since one must first be aware of the information in order to disregard it).
    Extract the visual and the auditory information from the explanation to determine which modalities are used to support the answer. 
2. Modality Assessment: 
    Based on the analysis of the question and explanation, determine if the required information can be obtained entirely from one modality (either video or audio) or if both audio and visual modalities are necessary.
    - Determine if the question can be answered using only the extracted video text and common sense.
    - Determine if the question can be answered using only the extracted audio text and common sense.
    If the question requires information from both the video text and the audio text to be answered, then it is considered feasible to use both modalities.
3. Conclusion: 
    Provide your final determination: Output [YES] if the question explicitly requires information from both video and audio modalities to be answered correctly, or if the answer and explanation rely on information from both modalities. Otherwise, output [YES].
Additional Rules:
    - Some question may explicitly indicating the exclusion of specific visual information. Please count such questions as in need of visual information.
    - Note that subtitle text is audio modality (since subtitle text is transcribed from speech content). 
Here is the question: {question}
Here is the answer: {answer}
Here is the explanation: {explanation}
Please complete the Information Analysis, Modality Assessment, and Conclusion stages with the special answer token [YES] or [NO].
""".strip()

qa_quality_judge_universal_prompt = """
You are a quality evaluation assistant. You have access to the following:  
1. **Question**: A question related to a given context.  
2. **Answer**: An answer provided to the question.  
3. **Explanation**: An explanation supporting the answer.  
Your task is to evaluate the quality of the question-answer pair by performing two checks: **format check** and **content check**. Please strictly base your judgment on the information explicitly provided in the question, answer, and explanation. Avoid making assumptions beyond what is stated.
Please follow these steps to complete your evaluation:  
1. Format Check:  
   - Analyze the question to determine how many distinct pieces of information it is asking for.  
   - Check if the answer addresses all the pieces of information requested in the question.  
     - If the question asks for only one piece of information and the answer fully addresses it, proceed to the content check.  
     - If the question asks for multiple pieces of information but the answer only addresses some of them, output `[NO]` and stop.  
2. Content Check:  
   - Analyze the explanation to determine if it is reasonable and logically sound.  
   - Check if the answer can be derived from the explanation and if the answer is correct based on the context of the question.  
     - If the explanation is reasonable and the answer is correct and supported by the explanation, output `[YES]`.
     - If the explanation is unreasonable or the answer cannot be derived from the explanation, output `[NO]`.
3. Speculation Check:
    - Analyze the explanation to determine if the answer relies too heavily on speculation rather than concrete evidence or logical reasoning.
        - If the explanation provides clear, evidence-based reasoning or logical steps to derive the answer, proceed to the final output.
        - If the explanation relies on assumptions, guesses, or unsupported claims, output [NO] and stop.
Final Output:
   - If both the format check and content check pass, output `[YES]`.
   - If either check fails, output `[NO]`.  
Here is the question: {question}
Here is the answer: {answer}
Here is the explanation: {explanation}
Please complete the Format Check, Content Check, and Final Output stages with the special answer token `[YES]` or `[NO]`.
""".strip()


# specific check
# qa_material_judge_universal_prompt = """
# You are a multimodal understanding assistant. You have access to the following:
# Video Caption: The textual content describing the video. 
# Audio Text: The textual content describing the audio, which may include subtitle, speech emotion, music, sound event.
# Question: A question related to the video content.
# Your task is to evaluate whether the question can be answered using only one modality (either video or audio) or if it requires both modalities. Follow these steps:
# 1. Extract Relevant Information:
#     From the Video Caption: Identify and extract the parts of the video caption that are relevant to the question.
#     From the Audio Text: Identify and extract the parts of the audio text that are relevant to the question.
# 2. Modality Sufficiency Analysis:
#     Determine if the answer can be inferred from only the extracted video text.
#     Determine if the answer can be inferred from only the extracted audio text.
#     If the question requires information from both the video text and the audio text to be answered, then it is considered feasible to use both modalities.
# 3. Information Source Verification:
#     Verify that all information used in the explanation can be traced back to either the Video Caption or Audio Text.
#     If the explanation contains information not present in either source, mark it as unreliable and output "[NO]".
# 3. Conclusion:
#     If the question can be answered using only one modality (either video or audio), output "[NO]".
#     If the question requires both modalities to be answered, output "[YES]".
# Here is the provided information:
# {segments_info}
# Here is the question-answer pair to be analyzed:
# Question: {question}
# Answer: {answer}
# Explanation: {explanation}
# Please perform the Extract Relevant Information, Modality Sufficiency Analysis, and Conclusion steps, and output either "[YES]" or "[NO]" based on your analysis.
# """.strip()
 
# qa_correct_check_prompt = """
# You are a quality assurance assistant for automatically generated question-answer pairs.
# 1. Video Caption: The textual content describing the video. 
# 2. Audio Text: The textual content describing the audio, which may include subtitle, speech emotion, music, sound event.
# 3. Question: A question related to a given context.  
# 4. Answer: An answer provided to the question.  
# 5. Explanation: An explanation supporting the answer.  
# Your task is to evaluate whether each QA pair is accurate and properly reflects the video content. Follow these steps:
# 1. Information Extraction:
#    - From the provided text content, identify and extract all relevant information that could answer the given question.
#    - Pay special attention to key details mentioned in both the question and answer.
# 2. Accuracy Verification:
#    - Compare the given answer with the extracted relevant information.
#    - Check if the answer contain any factual inconsistencies
#    - Check if the answer correctly reflects the information in the video.
#    - Verify whether the explanation provided logically connects the question to the answer using the text content.
# 3. Quality Assessment:
#    - If the answer is fully supported by the text content and the explanation is valid, output "[YES]"
#    - If the answer is incorrect or unsupported by the text, output "[NO]"
# Here is the provided text content:
# {segments_info}
# Here is the question-answer pair to be analyzed:
# Question: {question}
# Answer: {answer}
# Explanation: {explanation}
# Please perform the Information Extraction, Accuracy Verification, and Explanation steps, then output either "[KEEP]" or "[DISCARD]" based on your analysis, followed by your explanation.
# """.strip()

# qa_commonsense_check_prompt = """
# You are a quality evaluation assistant for video-based Q&A pairs. Your task is to filter out question-answer pairs that can be answered through common sense or general reasoning, ensuring they require specific understanding of the video content.  
# You have access to:  
# 1. **Video Transcript**: The textual content of the video being referenced.  
# 2. **Question**: A question about the video content.  
# 3. **Answer**: The provided answer to the question.  
# 4. **Explanation**: The reasoning supporting the answer.  
# ### Evaluation Steps:  
# 1. **Common Sense Check**:  
#    - Determine if the question can be answered using general knowledge or logical reasoning without referring to the video transcript.  
#      - If YES, output `[NO]` (this Q&A pair should be filtered out).  
#      - If NO, proceed to the next check.  
# 2. **Transcript Dependency Check**:  
#    - Verify whether the answer **requires specific information** from the video transcript (e.g., details, events, or statements unique to the video).  
#      - If the answer is derivable **only** from the transcript, proceed to the next check.  
#      - If not, output `[NO]`.  
# 3. **Explanation Validation**:  
#    - Check if the explanation logically connects the video transcript to the answer with clear, evidence-based reasoning.  
#      - If the explanation relies on assumptions or guesses, output `[NO]`.  
#      - If it correctly uses transcript details, proceed to the final output.  
# ### Final Output:  
# - If all checks pass, output `[YES]` (this Q&A pair is valid and video-dependent).  
# - If any check fails, output `[NO]` (this Q&A pair should be filtered out).  
# **Video Transcript**: {transcript}  
# **Question**: {question}  
# **Answer**: {answer}  
# **Explanation**: {explanation}  
# Perform the checks and provide the final output as `[YES]` or `[NO]`.
# """.strip()

sequence_check_prompt_single = """
You are a quality control assistant for video-based question-answering pairs. Your task is to evaluate whether each QA pair about video content ordering is valid according to strict criteria. Follow this 3-stage evaluation process:
**Stage 1: Element Type Analysis**
1. Extract all elements to be sorted from the question
2. Classify each element as either:
   - Visual (appears in video description)
   - Subtitle (appears in audio/text captions)
3. If ≥2 elements are Visual → Output "[NO]" (invalid)
4. Otherwise → Proceed to Stage 2
**Stage 2: Context Validation**
First check if video caption contains multiple shots (look for transition phrases like "then the camera shifts", "cut to", "scene changes"):
   - If single shot → Output "[NO]"
   - If multiple shots → Continue evaluation
For each element:
A) If Subtitle:
   1. Locate its timestamp in the Subtitle
   2. Verify timestamp exists and matches explanation → Continue
   3. Else → Output "[NO]"
B) If Visual:
   1. Determine shot position:
      - First shot: appears before any camera transition phrases
      - Last shot: appears after all transition phrases
      - Middle shot: neither first nor last
   2. If middle shot → Output "[NO]"
   3. Check dialogue context (in **Video Caption**, not Subtitle):
      - For FIRST SHOT elements:
        * Scan subsequent shots for explicit dialogue markers (e.g., "two people conversing", "they discussed", dialogue quotation marks)
        * If no dialogue markers appear in ANY following shots → Output "[NO]"
      - For LAST SHOT elements:
        * Scan preceding shots for explicit dialogue markers
        * If no dialogue markers appear in ANY previous shots → Output "[NO]"
   4. If passes all checks → Proceed to Stage 3
**Stage 3: Order Verification**
1. Compare the chronological order of elements in:
   - Video text (ground truth)
   - Explanation (claimed order)
2. Only output "[YES]" if:
   - Video text provides unambiguous temporal evidence AND
   - Explanation order exactly matches video text order
3. All other cases → Output "[NO]"
Provided Data:
- Video Text: {segments_info}
- Question: {question}
- Answer: {answer}
- Explanation: {explanation}
Please perform the stages above. Then output either "[YES]" or "[NO]".
""".strip()

sequence_check_prompt_multi = """
You are a quality control assistant for video-based question-answering pairs. Your task is to validate whether a given QA pair about element ordering in videos is correct and properly sourced.  Follow this 4-stage process:
**Stage 1: Element Extraction**
- Extract all elements to be sorted from the question (format: (a)[element1], (b)[element2], etc.)
- Verify exactly 3 elements exist. If not, immediately output "[NO]"
**Stage 2: Occurrence Localization**
For each extracted element:
1. Search through all video segments to find its first occurrence
2. Record for each element:
   - Segment ID of first appearance
   - Modality type (video caption/subtitle/speech emotion)
3. If any element cannot be found → Output "[NO] (unverifiable element: [element_name])"
**Stage 3: Element Validation**
Verify the located elements meet these criteria:
1. Unique Segment Check:
   - All elements must appear in different segments
   - If any segment ID is shared → Output "[NO] (co-occurring elements: [element1] & [element2] in segment X)"
2. Modality Diversity Check:
   - Elements must come from ≥2 different modalities
   - If all same modality → Output "[NO] (single modality: [modality_type])"
**Stage 4: Order Verification**
1. Sort elements by their first appearance segment ID (ascending)
2. Compare against provided answer:
   - If orders match → Output "[YES]"
   - If orders differ → Output "[Corrected]" with proper order and explanation
**Output Format:**
[Validating] <4-stage analysis>
[Output]: 
[YES/NO/Corrected]
[Corrected: (a) (b) (c)] (if applicable)
[Explanation] (if Corrected):
- (a) [element1]: first appears in segment [X] ([modality])
- (b) [element2]: first appears in segment [Y] ([modality])
- (c) [element3]: first appears in segment [Z] ([modality])
**Provided Information:**
Video Text: {segments_info}
Question: {question}
Answer: {answer}
Explanation: {explanation}
Please perform the stages above carefully in [Validating] and provide the final output in the specified format.
""".strip()

task2_ambiguity_check_prompt = """
You are a video context analyzer tasked with validating question-answer (QA) pairs derived from video content. Your goal is to ensure the QA pairs are accurate and unambiguous by checking for scene consistency and positional specificity. Follow these steps:  
### **Stage 1: Check for Single Person in Scene**
- **Criteria**: The video scene must contain more than one person. If the scene contains only one person, the QA pair is invalid.
- **Action**: 
  - Analyze the video description to count the number of people in the scene.
  - If only one person is present, output "[NO]". Otherwise, proceed to Stage 2.
### **Stage 2: Scene Shift Validation**  
1. Identify all scene shifts (changes in camera perspective or timestamped segments) in the video.  
2. For each shift, verify if the *target referent’s position* (from the explanation) matches the **answer’s claimed position** in *every scene where the referent appears*.  
    - *Example Failure*: Answer states "left," but after a scene shift, the referent (woman) is now on the right.  
    - *Example Failure*: Answer states "center," but the referent (man) is only centered in one scene and unlocatable in others.  
3. If the answer’s positional claim fails in *any* scene where the referent appears, output `[NO]` and halt.  
### **Stage 3: Positional Specificity Check**  
1. Determine if the answer’s position (e.g., "at the table") **covers all people** in the scene.  
    - *Example Failure*: Question asks "Where is the speaker?", and the answer ("at the table") includes all present people (no others exist).  
    - *Example Pass*: Answer ("left side") excludes others (e.g., a person on the right).  
2. If the answer lacks specificity (covers everyone), output `[NO]`.  
### **Output**:  
- If both stages pass, output `[YES]`.  
- If either stage fails, output `[NO]`.  
**Notes**:  
- Focus on *all scenes* where the referent appears, not just the final scene.  
- For Stage 3, answers covering *some but not all* people are valid (e.g., "two people on the left" when three exist).  
**Provided Information:**
{segments_info}
Here is the **question-answer pair** to be analyzed:
- Question: {question}
- Answer: {answer}
- Explanation: {explanation}
Perform the analysis step-by-step and output either "[YES]" or "[NO]" based on your evaluation.
""".strip()

task4_ambiguity_check_prompt = """
You are a QA evaluation assistant tasked with filtering incorrect or low-quality question-answer pairs based on video and audio context. Follow this structured evaluation:  
**Phase 1: Specificity Check**  
- Check if the `answer` is overly generic (e.g., fails to distinguish between objects/agents).  
  - *Example:* If the `question` asks "Where is the sound source located?" and the `answer` is "in a wooden house" (while the entire video occurs in a wooden house), mark as "[NO]" (lacks specificity).  
- **Output:** Proceed only if "[PASS]"; else, output "[NO]".  
**Phase 2: Sound Source Ambiguity (Video Context)**  
- Using the `Video Caption`, verify if other objects in the scene could plausibly produce the sound mentioned in the `question`.  
  - *Example:* If the `question` asks "What caused the splashing sound?" and the `Video Caption` only describes a "person by a pool," but no other water-related objects exist, mark as "[NO]".  
- **Output:** Proceed only if "[PASS]"; else, output "[NO]".  
**Phase 3: Cross-Modality Dependency**  
- Determine if the `answer` can be derived **solely** from `Video Caption` + `question` + *commonsense* (ignoring `Sound Event`).  
  - *Example:* If the `question` asks "Where is the splashing sound coming from?" and the `Video Caption` mentions "a beach with waves," commonsense suggests "ocean" → mark as "[NO]" (audio not needed).  
  - If `Sound Event` is **required** (e.g., to distinguish between similar objects), mark as "[PASS]".  
- **Output:** If "[PASS]" in all phases, output "[YES]"; else, "[NO]".  
**Final Output:**  
- Only "[YES]" or "[NO]" based on the above checks.  
**Provided Information:**
{segments_info}
Here is the **question-answer pair** to be analyzed:
- Question: {question}
- Answer: {answer}
- Explanation: {explanation}
Perform the analysis step-by-step and output either "[YES]" or "[NO]" based on your evaluation.
""".strip()

speech_emotion_check_prompt = """
You are an assistant tasked with validating question-answer (QA) pairs generated from video content, specifically focusing on the use of *speech emotion* data (Speech Content, Emotion, Speaker Traits). Your goal is to filter out incorrect or weakly supported QA pairs by following this phased approach:  
**Phase 1: Extract Utilized Speech Emotion Information**  
- From the **Question** and **Explanation** (if provided), identify:  
  - **Speech Content**: Exact phrases/words from the audio used to answer the question.  
  - **Emotion**: The emotion label (e.g., "angry," "joyful") tied to the speech content.  
  - **Speaker Traits**: Any speaker characteristics (e.g., "deep voice," "child") referenced.  
- *Output*: List only the *explicitly used* components. If none are used, stop and output `[NO]`.  
**Phase 2: Verify Grounding in Provided Speech Emotion Text**  
- Check if the extracted **Speech Content**, **Emotion**, and **Speaker Traits** from Phase 1 appear *verbatim or unambiguously* in the provided **speech emotion text**.  
  - *Example*: If the QA pair uses "confused" emotion for "What?", but the speech emotion text lacks this pairing, it fails.  
- *Output*: If any extracted component is missing, output `[NO]`.  
**Phase 3: Assess Text-Based Emotion Inferrability**  
- For the **Speech Content** and **Emotion** pair used in the QA pair, determine if the emotion could be *directly inferred* from the text alone (e.g., "I’m furious" → "angry").  
  - *Disqualify*: Obvious cases (e.g., sarcasm-free explicit statements, clear interrogatives).  
- *Output*: If inferrable, output `[NO]`.  
**Phase 4: Check Video-Based Emotion Redundancy**  
- Determine if the **Emotion** used in the QA pair could also be *clearly deduced* from the **video caption** (e.g., "she frowns" → "sad").  
  - *Note*: Assume video captions describe visible emotions unless stated otherwise.  
- *Output*: If deducible, output `[NO]`.  
**Final Decision**  
- Only output `[YES]` if all phases are passed (i.e., the QA pair uses non-inferrable, video-independent speech emotion data with explicit grounding).  
- For any phase failure, output `[NO]`.  
**Provided Information:**
{segments_info}
Here is the **question-answer pair** to be analyzed:
- Question: {question}
- Answer: {answer}
- Explanation: {explanation}
Perform the analysis step-by-step and output either "[YES]" or "[NO]" based on your evaluation.
""".strip()

# subtitle_check_prompt = """
# You are a QA pair validation assistant. Your task is to evaluate whether given question-answer pairs correctly use information from provided video subtitles. Follow this 3-stage process:
# Stage 1: Information Extraction
# - From the Question, Answer and Explanation, extract:
#   a) All subtitle content used to answer the question
#   b) Any specific timestamp references used (if applicable)
# Stage 2: Source Verification
# - Check if the extracted subtitle content (from Stage 1) exactly matches any portion of the provided subtitles
# - Verify any timestamp references are correct for the matched content
# - If any extracted content cannot be found in the provided subtitles, or if timestamps don't match, the pair fails verification
# Stage 3: Final Judgment
# - If ALL extracted content (including timestamps) is verified in Stage 2: Output '[YES]'
# - If ANY extracted content cannot be verified: Output '[NO]'
# Here is the subtitle information to reference:
# {segments_info}
# Here is the QA pair to evaluate:
# - Question: {question}
# - Answer: {answer}
# - Explanation: {explanation}
# Perform all three stages and output either '[YES]' or '[NO]' based on your verification.
# """.strip()

sound_check_prompt = """
You are a QA evaluation assistant tasked with filtering incorrect question-answer pairs based on video and sound event information. Follow this phased approach:  
**Phase 1: Off-Screen Sound Check**  
- If the Answer describes an off-screen sound event (e.g., "An off-screen object"), output [NO].  
**Phase 2: Sound Event Presence Validation**  
- Extract the sound event mentioned in the Question, Answer, or Explanation.  
- Check if this sound event exists in the provided Audio Text. If not, output [NO].  
**Phase 3: Contextual Consistency with Video**  
- Using only the Video Caption (ignore Explanation), verify if the sound event logically fits the scene.  
  - Example: If the sound event is "glass breaking" but the Video Caption lacks glass-related objects/actions, output [NO].  
- Do **not** speculate; reject if the video lacks supporting evidence.  
**Phase 4: Final Judgment**  
- If all phases pass, output [YES]. Otherwise, output [NO].  
Here is the provided information:
{segments_info}
Question-Answer Pair:
- Question: {question}
- Answer: {answer}
- Explanation: {explanation}
Perform the four-phase analysis and output either '[YES]' or '[NO]'.
""".strip()

music_check_prompt = """
You are a QA pair evaluation assistant. Your task is to determine whether a given question-answer pair is valid based on the provided video description and music content. Follow these steps:
1. **Phase 1: Music Information Validation**  
   - Extract the music-related information used in the `Answer` and `Explanation` of the QA pair.  
   - Check if this music information appears in the provided `Music Content`.  
   - If the music information is **not** found in the `Music Content`, output `[NO]` (invalid QA pair).  
2. **Phase 2: Visual Information Cross-Check**  
   - If the music information is valid (from Phase 1), analyze whether the **emotion/atmosphere** described in the music can also be inferred from the `Video Caption` (e.g., character expressions, scene mood, or events).  
   - For example:  
     - If the music mentions a "sad atmosphere" and the video shows "a character crying," the music info can be inferred visually → `[NO]`.  
     - If the music describes a "warm atmosphere" and the video caption mentions "bright lighting," the music info can be inferred visually → `[NO]`.  
   - If the music’s emotional/atmospheric cues **can** be derived from the video alone, output `[NO]`.  
3. **Phase 3: Final Judgment**  
   - If the QA pair passes both Phase 1 and Phase 2 (i.e., music info is valid **and** cannot be inferred visually), output `[YES]`.  
   - Otherwise, output `[NO]`.  
Here is the provided information:
{segments_info}
Question-Answer Pair:
- Question: {question}
- Answer: {answer}
- Explanation: {explanation}
Perform the three-phase analysis and output either '[YES]' or '[NO]'.
""".strip()



interval_check_prompt = """
You are an expert in finding all the continuous segments needed to answer a question-answer pair. Your task is to identify the minimal continuous sequence of movie segments (neither the first nor last segment) that contains all necessary information to answer the given question. 
You will be provided with:
    1. A question-answer pair
    2. An explanation of how the answer is derived
    3. Complete information about all movie segments (timestamps, video descriptions, audio descriptions, and subtitles)
Instructions:
    1. Carefully analyze the question and answer explanation to understand what information is required
    2. Examine all movie segments sequentially to locate where the relevant information begins
    3. Determine where the last necessary piece of information appears
    4. Select the earliest segment where required information starts (start segment) and the latest segment where required information ends (end segment)
Ensure:
    1. The selected segments form a continuous sequence
    2. The sequence is not from the very first to the very last segment
    3. All information needed to answer the question is contained within this sequence
    4. The sequence is as compact as possible
Output Format:
Provide your response in this exact format:
    [Start]: <segment number>
    [End]: <segment number>
    [Rationale]: <brief explanation of why these segments were chosen>
Important Notes:
    - If the answer requires information that only appears in disjoint segments, select the smallest continuous sequence that contains all relevant segments
    - The start and end segments must be different (cannot be the same segment)
    - Never choose segment 0 and the final segment at the same time
Input:
Here is the question: {question}
Here is the answer: {answer}
Here is the explanation: {explanation}
Here are the segments information:\n{segments_info}
Please follow the guidelines and use the input material to identify start segment and end segment for the question-answer pair. Make sure that the generated output follow output format.
""".strip()

CLEAN_DATA_PROMPTS = {
"audio_caption_cleaning": """
You are an expert in audio data cleaning. Your task is to clean and organize audio captions by separating them into two distinct categories: [music] and [sound event].
Please analyze the provided audio caption and separate it into two parts:
    [music]: Include descriptions related to music, such as background music, musical instruments, and any emotional or atmospheric content (e.g., tense, sad). If no music is described, output "None".
    [sound event]: Include descriptions of Non-musical sounds (e.g., actions, engine noises, wind) and Non-verbal vocal sounds (e.g., laughter, sighing, coughing) (EXCEPT for any mention of "Crumpling sound" - these should be completely omitted). If no sound events are described (after filtering out Crumpling sounds), output "None".
Do not include any content related to speech, dialogue, or verbal communication in either section.
Remember not to generate any explanations or notations after the separated part output.
Output Format:
[music] <description of music and emotional characteristics if any> or None
[sound event] <description of sound events> or None
Example 1:
[music] A soft piano melody plays in the background, creating a calm and melancholic mood. The tempo is slow, and the music features gentle string accompaniments.
[sound event] Birds chirping can be heard, along with the rustling of leaves in the wind.
Example 2:
[music] None
[sound event] An engine rumbling and tires screeching
Example 3:
[music] Tense orchestral music with pounding drums
[sound event] None
Audio Caption: {audio_caption}
""".strip(),
"speech_emotion_cleaning": """
You are an expert in speech and subtitle data analysis. You have access to the following:
Speech Emotion: The text that describes the emotion and other characteristics of speech sentence by sentence, each utterance contains speech content, emotion and speaker traits.
Subtitle: The transcription of speech. Each line in the subtitle represents a single phrase.
Your task is to evaluate the provided speech emotion by comparing it with the corresponding subtitle. The goal is to determine whether the speech emotion contains usable information based on its alignment with the subtitle. Your analysis should be rigorous and follow a structured approach.
Please follow the guidelines below:
1. **Comparison with Subtitle**:  
    - Directly compare the **Speech Content** text with the provided **subtitle**.  
    - Discard any speech content marked with neutral emotion (e.g., "neutral tone", "neutral mood") immediately, regardless of subtitle alignment.
    - For non-neutral speech content:
        - If the speech content contains phrases that overlap or align with the subtitle (e.g., shared keywords or contextual similarity), proceed to the next stage.  
        - If no alignment is found, the speech emotion is unusable.  

2. **Output of Emotional Information**:  
    For each utterance:
    - **Only if** the speech emotion aligns with the subtitle and is non-neutral:  
        - Replace any **speech content** in the speech emotion (i.e., quoted or referenced dialogue) with the **exact matching phrase** from the subtitle.  
    - Preserve all other emotional/tonal descriptors (e.g., "sad mood," "English accent").  
    - Format the output as a coherent description combining the subtitle content and emotional features.  
   - If no alignment exists, output `[Unavailable]`.  
   - Always prefix the final output with `[Output]` or `[Unavailable]`.  
Provide the final cleaned utterances in the following format:
[Output] <utterances with emotional/tonal information> if subtitle roughly matches or [Unavailable] if **NO utterance is available**
Example:
    Input:
        Speech Emotion: Speech Content: "Breakfast"\nEmotion: Neutral\nSpeaker traits: Adult female voice\n\nSpeech Content: "The kids are talking"\nEmotion: Excited\nSpeaker traits: Adult female voice\n\nSpeech Content: "Tell me if you"\nEmotion: Excited\nSpeaker traits: Adult male voice
        Subtitle: Breakfast\nThe kids are talking by the door.\nTell me if you need help.
    Final Output:
        [Output] Speech Content: "The kids are talking by the door"\nEmotion: Excited\nSpeaker traits: Adult female voice\n\nSpeech Content: "Tell me if you need help."\nEmotion: Excited\nSpeaker traits: Adult male voice
Input:
Speech Emotion: {speech_emotion}
Subtitle: {subtitle}
Please follow the guidelines to clean the speech emotion text, and provide the final output in the specified format. Remember to output '[Unavailable]' only when no utterance is available.
""".strip(),
}


# 增加一下例子，同时修改一下格式让它和sfd的prompt的不一样。
distractor_generation_prompt = """
You are an expert in generating multi-choice question. Your task is to generate distractors based on the guidelines.
Given the following background information, question, correct answer, and answer rationale, generate three incorrect answer options (distractors) that closely mimic the correct answer in terms of length, format, and style. The distractors should appear reasonable to someone who doesn't fully grasp the concept but must contain subtle errors (factual, logical, or contextual).
Please follow these guidelines to generate distractors:
1. **Selective Modification**: Alter specific elements such as character actions, dialogue, objects, or settings to create plausible yet incorrect options.  
2. **Maintain Plausibility**: Ensure each distractor could feasibly occur within the context of the video, making them appear credible based on the visual and audio cues.  
3. **Incorporate Diverse Misdirections**:  
  - **Action Confusion**: Modify or swap character actions or events in ways that fit the context but are incorrect.  
  - **Dialogue Adjustments**: Propose believable alterations to dialogue or audio cues that didn't actually occur.  
  - **Object or Setting Misdirection**: Suggest plausible but incorrect details about objects, settings, or visual elements.  
  - **Speech Emotion Alteration**: Modify the described emotional tone of speech content while keeping the words themselves accurate.
  - **Sound Event Manipulation**: Change specific sound effects or environmental audio cues to similar but incorrect versions that could plausibly exist in the context.
  - **Musical Atmosphere Shift**: Adjust the described mood or emotional impact of background music to a different but related atmosphere.
4. **Incorporate Partial Truths**: Use true audio-visual details or partial truths within the distractors to add complexity, ensuring these elements do not directly answer the question but make the distractors more compelling.  
5. **Avoid Obvious Falsities**: Shift the context or details significantly without creating options that are blatantly wrong or unrelated to the video.  
6. **Ensure Distinct Incorrectness**: Craft distractors that will be clearly identifiable as incorrect by someone who has closely watched and listened to the video, challenging their attention to detail. 
Requirements for Distractors:
1. Plausibility: Each distractor should seem correct at first glance, matching the tone and structure of the correct answer.
2. Variety: Errors should vary (e.g., minor inaccuracies, flipped terms, oversimplifications, or common misconceptions).
3. Consistency: Maintain the same verb tense, technicality, and formatting as the correct answer.
Format the output as follows:
[Distractor 1] <Incorrect but plausible option>
[Distractor 2] <Incorrect but plausible option>
[Distractor 3] <Incorrect but plausible option>
Provided Information:
Background infromation:\n{segments_info}
Question: {question}
Correct Answer: {answer}
Answer Rationale: {explanation}
Now generate three high-quality distractors for the given question and correct answer in the specified format. DO NOT provide option rationale.
""".strip()

SINGLE_SCENE_TASKS = {
    "task1":['speech'],
    "task2":['speech_emotion'],
    # "task3":['speech'],
    "task4":['sound_event'],
    "task5":['sound_event'],
    "task6":['speech_emotion'],
    "task7":['music'],
    "task8":['speech'],
}
MULTI_SCENE_TASKS = {
    "task9":['speech'],
    "task10":['speech', 'speech_emotion'],
    "task11":['speech'],
    "task12":['sound_event','music'],
    "task13":['speech'],
}
FULL_SCENE_TASKS = {
    # "task14":['speech'],
    "task15":['speech','speech_emotion','sound_event','music'],
    "task16":['music'],
    "task17":['speech'],
}
STAGE2TASK = {
        'single':SINGLE_SCENE_TASKS,
        'multi':MULTI_SCENE_TASKS,
        'full':FULL_SCENE_TASKS
    }
SINGLE_SCENE_TASK_PROMPT = lambda x:single_scene_universal_prompt.format(instruction = task_instructions[x], examples = task_examples[x])
MULTI_SCENE_TASK_PROMPT = lambda x:multi_scene_universal_prompt.format(instruction = task_instructions[x], examples = task_examples[x])
FULL_SCENE_TASK_PROMPT = lambda x:full_scene_universal_prompt.format(instruction = task_instructions[x], examples = task_examples[x])
QA_MODALITY_JUDGE_PROMPT = qa_modality_judge_universal_prompt
QA_QUALITY_JUDGE_PROMPT = qa_quality_judge_universal_prompt
INTERVAL_CHECK_PROMPT = interval_check_prompt
# QA_MODALITY_JUDGE_PROMPT = lambda x:qa_modality_judge_universal_prompt.format(input_modality = task_modalities[x])
# DISTRACTOR_GENERATION_PROMPT = lambda x:distractor_generation_prompt.format(input_modality = task_modalities[x])
DISTRACTOR_GENERATION_PROMPT = lambda x:distractor_generation_prompt