from transformers import pipeline

# Load the summarizer pipeline
summarizer = pipeline("summarization")

# Long input text
input_text = '''We play the game at recess, and the teachers don’t notice. We stand on the playground by the 
               flagpole, arms ringed with colored bracelets from the drugstore, waiting. The boys come past us,
               in a bunch, all elbows, laughing. They pretend not to look. We pretend not to see them. One of 
               them reaches out and snaps a bracelet off one of us, breaking it like a rubber band, fast and 
               sharp as plucking a guitar string. He won’t look back. He’ll walk back the way he came, along 
               the edge of the football field. And whoever he picked, Angie or Carrie or Mandy, will watch 
               him go. After a minute she’ll follow him and meet him under the bleachers, far down the field, 
               where the teachers can’t see.'''

# Get the summary
summary = summarizer(input_text, max_length=60, min_length=20, do_sample=False)

# Display results
print("Original text:\n", input_text)
print("\nSummary:\n", summary[0]['summary_text'])
