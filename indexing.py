def search_transcript(transcript, query):

    results = []

    sentences = transcript.split(".")

    for s in sentences:

        if query.lower() in s.lower():

            results.append(s)

    return results