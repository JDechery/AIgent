import pickle
from gensim import models
from Mediumrare import gensim_nlp, db_tools, predictor_model, channel_metadata
import numpy as np
import pudb

def ModelIt(blogtext, embedder_model, clf, topN=5):
    _,_,labelencoder, channeldf = predictor_model.reorg_for_training(embedder_model, min_blogs=25)

    blog_tokens = db_tools.clean_document(blogtext)
    blog_vec = embedder_model.infer_vector(blog_tokens).reshape(1,-1)

    class_probs = clf.predict_proba(blog_vec)[0].tolist()
    top_idx = np.argsort(class_probs)[:topN]
    top_channels = labelencoder.classes_[top_idx]

    channel_data_funs = (channel_metadata.most_recent_pubs,
                         channel_metadata.most_common_tags,
                         channel_metadata.mean_channel_claps)
    top_channel_data = [[data_fun(channeldf, chan) for data_fun in channel_data_funs] for chan in top_channels]
    # top_channel_data.append([channel_metadata.most_similar_doc(channeldf, chan, blog_vec) for chan in top_channels])
    # pudb.set_trace()
    return top_channel_data
# %%
def ModelBetter(blogtext, embedder_model, clf, topN=5):
    # blogtext, embedder_model, clf = get_debug_vars()
    _,_,labelencoder, channeldf = predictor_model.reorg_for_training(embedder_model, min_blogs=25)

    blog_tokens = db_tools.clean_document(blogtext)
    blog_vec = embedder_model.infer_vector(blog_tokens).reshape(1,-1)

    class_probs = clf.predict_proba(blog_vec)[0].tolist()
    top_idx = np.argsort(class_probs)[-1:-topN-1:-1]
    # top_idx = np.argsort(class_probs)[-topN:]
    top_channels = labelencoder.classes_[top_idx]
    channel_names = list(map(lambda x: x.replace('-',' ').title(), top_channels))
    channel_names = ['The Washington Post' if x=='thewashingtonpost' else x for x in channel_names]
    channel_url = ['https://medium.com/' + x for x in top_channels]
    top_probs = np.asarray(class_probs)[top_idx]

    relevances = top_probs / np.sum(top_probs)
    relevances = [str(round(x*1000)/10) for x in relevances]

    similar_pubs = [channel_metadata.most_similar_doc(channeldf, topchan, blog_vec, embedder_model) for topchan in top_channels]
    recent_pubs = [channel_metadata.most_recent_pubs(channeldf, topchan) for topchan in top_channels]
    tagdf = db_tools.channel_tag_df()
    tagdf.set_index('channel', inplace=True)
    taglines = tagdf.loc[top_channels].values
    taglines = [tag[0] for tag in taglines]
    article_tags = [channel_metadata.most_common_tags(channeldf, topchan) for topchan in top_channels]
    article_tags = [[x for x in tags if x] for tags in article_tags]
    top_channel_data = (relevances,
                        tuple(zip(channel_names, channel_url, taglines)),
                        article_tags,
                        similar_pubs,
                        recent_pubs)
    top_channel_data = list(zip(*top_channel_data))
    return top_channel_data
# %%
def get_debug_vars():
    embedder = gensim_nlp.DocEmbedder()
    embedder.load_model()
    clf, *_ = predictor_model.load_classifier()
    blogtext = '''How To Change Your Life In 30 Days
    Your identity is not fixed, but highly fluid.
    Your identity follows your behaviors.
    How does this work?
    It works based on two very important psychological concepts:
        Self-signaling: which means that, as a person, you evaluate and judge yourself the same way you judge others — based on behavior. So, if you watch yourself do something, you identify yourself with that behavior. If you drink alcohol, for example, you begin to identify yourself as someone who drinks alcohol. If you wake up early, you identify yourself as someone who wakes up early. If you write articles online, you identify yourself as a writer. Thus, how you see yourself is highly fluid, and based on your own behaviors. As your behavior changes, your perceived identity changes.
        Precognition: which means that thoughts don’t necessarily lead to behaviors, but that behaviors can also lead to thoughts. In other words,common wisdom suggests that your inner world creates your outer world.Hence, “mental creation precedes physical creation.” This is certainly true. But behaviors (and environments) can also create internal states. For example, if you jump into an ice-cold bath, you’ll begin to experience a cascade of emotions and thoughts. Or lack of thoughts. What precognition shows is that you can actually PREDICT your inner state by behaving in certain ways, and by placing yourself in certain environments. Thus,change doesn’t only happen from the inside out, but also from the outside in.
    Both of these ideas are strongly related to other research in psychology, which suggests that behaviors generally come BEFORE psychological states. Again, this goes against most common wisdom.
    My favorite example is the research on self-efficacy (confidence), which shows that confidence isn’t what produces high performance. But rather, that high performance is what produces confidence.
    Put simply, if you want to have confidence, you can have it. All you have to do is behave in desired ways, even for a short period of time.
    Why does all of this matter?
    It matters, because you have the power to radically change your identity.
    Even at a biological level, new science in epigenetics and neuroplasticity is showing how malleable and fluid our biology is.
    The Problem With Succeeding
    Most people plateau.
    Even successful people.
    It’s actually very common for people who are succeeding to get stuck.
    Think about some of your favorite authors, musicians, and even companies.
    At some point, they generally stop being as innovative.
    We all have that band we love, whose first album or two had way more soul.Then, once they became famous, their music became far more tame.
    The same is often true of world-class chefs.
    Once a restaurant becomes highly successful, they usually stop innovating the menu as much.
    Once something is working, it’s hard to go back to ground zero.
    In psychological terms, your motivation can go from approach-oriented to avoid-oriented.
    Specifically, all goals are either offensive or defensive.
    If you’re seeking to advance your current position, you’re “approaching.”
    If you’re seeking to maintain your position, or avoid something bad from happening, you’re “avoiding.”
    When you’re approaching, you’re less concerned about risks and more focused on rewards. You’re willing to take risks. You’re willing to fail. You’re being PULLED forward.
    When you’re avoiding, you’re less concerned about the rewards and more focused on the risks. And you have no desire to proactively confront those risks. Instead, you’re simply trying to shield yourself from any problems that come your way.
    I’ve seen this with many of my role models. For example, some of my favorite authors have shifted from approach-oriented to avoid-oriented.
    I can see it in their work.
    It’s become far more safe.
    They are making far less significant ideological attempts in their writing.Their books are becoming more mainstream. Obviously calculated and less intuitive and inspired.
    When you begin succeeding, your focus can shift from WHY to WHAT. Instead of operating from your core, your simply try to maintain success.
    This is how you get stuck.
    This is how you get confused and lose your identity.
    Are you on offense or defense?
    Are you approaching or avoiding?
    Are you proactively becoming the person you want to be?
    Or are you holding on to the person you think you are?
    The Antidote: Never Stop Re-Inventing Yourself
    In the brilliant Netflix documentary, Chef’s Table, which highlights the lives of the world’s most successful chefs, one particular episode stands out.
    The number one chef in Asia, Gaggan Anand, is known for spontaneously throwing out his entire menu and starting from scratch. Even when his current menu is getting lots of attention.
    This may not seem like a big deal, but it is.
    When a restaurant starts getting recognized and certain awards, it’s generally based on the menu and overall atmosphere.
    Being literally number one in Asia, it would make sense for Gaggan to keep his restaurant how it is.
    But that’s not what he does.
    Creativity, and always pushing his own boundaries, is what he is about.
    So just because something is working doesn’t give him permission to stop evolving.
    So he reinvents himself.
    Over and over and over.
    No matter how hard it is to walk away from something brilliant.
    A true creator never stops pushing their boundaries.
    They never stop reinventing.
    Once you become awesome at something, use your new LEARNING ABILITIES to become awesome at something else.
    The whole notion of “finding your calling” has led people to having fixed views of themselves.
    There isn’t just one thing you were born to do.
    You can expand and grow in countless ways. Especially after you learn the process of learning. You can take all of your experience becoming great at something, and quickly become proficient at something else.
    In this way, you never plateau. You’re always growing and evolving as a person.
    The 30-Day Challenge
    Given that your identity is fluid and malleable, you have an amazing opportunity to redefine who you are.
    All you have to do is consistently and boldly reshape your behavior.
    You can do this in the form of a 30-day challenge.
    What’s something you’ve wanted to do, that you haven’t done?
    Or, what’s something that would clearly lead you to a place you’d like to be?
    It could be 30 days of extreme health and fitness.
    That would definitely change things.
    It could be facing an extreme fear: like 30 days of asking people on dates.
    It could be 30 days of writing articles, or filming videos.
    Whatever it is, if you do it for 30 days, your identity will change.
    Your fears will become cauterized and neutralized.
    You’ll adapt to your new behaviors.
    Your psychological state will change.
    You’ll begin to identify with your new behaviors.
    Will you have to deal with some negative emotions along the way?
    Will you face a load of resistance and fear?
    Will you want to quit?
    The answer is probably yes to all of those questions.
    But THIS is how you separate yourself from the masses.
    This is how you make quantum leaps in your progression, while most people make incremental progress.
    This is how you consciously shape your identity and future.
    Where will you be in 30 days from now?
    WHO will you be 30 days from now?'''
    return blogtext, embedder.model, clf
