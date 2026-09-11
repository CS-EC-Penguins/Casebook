# My pipeline account

This has five sections, one per step of the pipeline. Fill in the first two boxes
of each as you work through the lab, and the third once you have made that decision on
the Casebook corpus. Two or three sentences per box is plenty.

Section 1 is the exception. There is no decision to make about putting text in a
prompt, so its third box asks for the number instead.

## 1. Putting the documents in the prompt

What it does: It tells the LLM which bit of text to analyse for a response, and allows for citations to happen.

What breaks without it: You can't cite what document is used, and you might be more likely to get a hallucination.

What one question costs if you send the whole EU AI Act with it, and what
happens to that number at two thousand documents: Input token cost is $0.3 per million tokens, or $0.0003 per thousand tokens. So if the 587,832 characters in the EU AI act consumes ~147k tokens, then the price for 1 call with the EU AI act included costs $0.0003*147 = $0.0441.

## 2. Cutting documents into chunks

Covers chunk size, overlap, and the separator list.

What it does: Splits large documents into smaller chunks. This allows savings to be made on token calls and making the llm analyse only the most relevant parts of a document. It also retains the ability to provide citations with answers.

What breaks without it: Without it you can get context drift and in-efficient tools that cost too much money to run.

What I chose for Casebook, and why:

## 3. Storing the vectors

What it does: Saves all the chunks in a database with their computed vectors and source metadata so that embeddings don't need to be re-calculated on every query.

What breaks without it: Without this, the ability to order by proximity in semantic meaning is taken away and the savings on LLM tokens are taken away because everything in the whole db would need to be re-analiysed and ordered for every new query/question on the document store.

What I chose for Casebook, and why: I would like to have extended metadata, expecially for the EU AI act that can be more specific about which part of the document is being referred to, as it's already chunked quite semantically.

## 4. Retrieving the closest passages

Covers how many passages you retrieve, and the source that travels with each one.

What it does:

What breaks without it:

What I chose for Casebook, and why:

## 5. Constraining the prompt

What it does:

What breaks without it:

What I chose for Casebook, and why:

## One more thing

If Casebook gave you an answer tomorrow that was wrong, or missing a citation,
which of the five would you look at first, and what would you print to check it?
