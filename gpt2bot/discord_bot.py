import asyncio
from .utils import *
import discord
import time
import random

class MyClient(discord.Client):
    def __init__(self, botObj):
        super().__init__()
        self.bot = botObj
        self.last_time = time.time()
        self.rec_messages = []
        self.priority_msg = []
        self.last_msg_id = 0
        self.channel_name = self.bot.discord.get("channel_name")
        self.min_time = float(self.bot.discord.get("delay", 1))

    def can_send(self):
        elapsed = time.time() - self.last_time
        if elapsed > self.min_time:
            self.last_time = time.time()
            return True
        else:
            return False

    def check_mention(self, m):
        mention = f'<@!{self.user.id}>'

        if m.reference is not None:
            return True
            #if self.last_msg_id == 0:
            #    self.last_msg_id = m.reference.message_id
            
            #if m.reference.message_id == self.last_msg_id:
            #    return True
        elif mention in m.content:
            return True
        return False

    async def on_ready(self):
        print('Logged in as '+self.user.name+' (ID:'+str(self.user.id)+') | Connected to '+str(len(self.guilds))+' servers')
        print('--------')
        print("Discord.py verison: " + discord.__version__)
        print('--------')

    async def on_message(self, message):
        # Only work on one channel
        # if not self.channel_name in message.channel.name:
        #     return
        if not message.channel.name == self.channel_name:
            return

        # Don't respond to ourselves
        if message.author == self.user:
            return
        
        # Do we have permission to talk in the channel?
        if self.get_effective_permissions(message.channel).send_messages == False:
            return

        # Append message to possible messages to answer list
        self.rec_messages.append(message.content)

        # Was the bot mentioned?
        if self.check_mention(message):
            print(f'BOT MENTION: {message.content}')
            self.priority_msg.append(message)

        # Is it time to send a message?
        if not self.can_send():
            return
        
        # Are there any messages to respond to?
        if len(self.rec_messages) == 0:
            return

        # Simulate typing a message
        async with message.channel.typing():
            # Are there any priority messages to process?
            if len(self.priority_msg) != 0:
                msg = self.priority_msg[0]
                self.priority_msg.clear()
                self.rec_messages.clear()
                response = self.bot.gen_message(msg.content, True)
                print(f'Answering user {message.author} WITH PRIO: {msg.content}')
                last_msg = await msg.reply(response, mention_author=False)
                self.last_msg_id = last_msg.id

            filtered = list(filter(lambda x: len(
                x) < 900 and '\n' not in x, self.rec_messages))  # filter long messages away  

            # Pick random message to answer
            to_answer = random.choice(filtered)
            self.rec_messages.clear()  # Clear the rec_messages
            print(f'Answering user {message.author}: {to_answer}')
            response = self.bot.gen_message(to_answer)
            last_msg = await message.channel.send(response)
            self.last_msg_id = last_msg.id

    def get_effective_permissions(self, channel):   
        """Get permissions that we have in a specific channel id"""  
        me = channel.guild.me

        granted_perms = me.guild_permissions

        role_list = [me] + me.roles

        for role in role_list:
            ovr = channel.overwrites_for(role)
            allow, deny = ovr.pair()

            granted_perms.value = granted_perms.value & (~deny.value)

            granted_perms.value = granted_perms.value | allow.value

        return granted_perms


# //////

class DiscordBot:
    def __init__(self, **kwargs):
        # Extract parameters
        general_params = kwargs.get('general_params', {})
        device = general_params.get('device', -1)
        seed = general_params.get('seed', None)
        debug = general_params.get('debug', False)

        generation_pipeline_kwargs = kwargs.get(
            'generation_pipeline_kwargs', {})
        generation_pipeline_kwargs = {**{
            'model': 'microsoft/DialoGPT-medium'
        }, **generation_pipeline_kwargs}

        generator_kwargs = kwargs.get('generator_kwargs', {})
        generator_kwargs = {**{
            'max_length': 1000,
            'do_sample': True,
            'clean_up_tokenization_spaces': True
        }, **generator_kwargs}

        prior_ranker_weights = kwargs.get('prior_ranker_weights', {})
        cond_ranker_weights = kwargs.get('cond_ranker_weights', {})

        chatbot_params = kwargs.get('chatbot_params', {})

        discord = kwargs.get('discord', {})

        self.generation_pipeline_kwargs = generation_pipeline_kwargs
        self.generator_kwargs = generator_kwargs
        self.prior_ranker_weights = prior_ranker_weights
        self.cond_ranker_weights = cond_ranker_weights
        self.chatbot_params = chatbot_params
        self.device = device
        self.seed = seed
        self.debug = debug
        self.discord = discord
        self.turns = []

        # Prepare the pipelines
        self.generation_pipeline = load_pipeline(
            'text-generation', device=device, **generation_pipeline_kwargs)
        self.ranker_dict = build_ranker_dict(
            device=device, **prior_ranker_weights, **cond_ranker_weights)

    def gen_message(self, msg, reply=False):
        """Receive message, generate response, and send it back to the user."""

        if reply:
            max_turns_history = 1
        else:
            max_turns_history = self.chatbot_params.get('max_turns_history', 2)

        user_message = msg

        if max_turns_history == 0:
            self.turns = []
        # A single turn is a group of user messages and bot responses right after
        turn = {
            'user_messages': [],
            'bot_messages': []
        }
        self.turns.append(turn)
        turn['user_messages'].append(user_message)
        # Merge turns into a single prompt (don't forget EOS token)
        prompt = ""
        from_index = max(len(self.turns) - max_turns_history - 1,
                         0) if max_turns_history >= 0 else 0
        for turn in self.turns[from_index:]:
            # Each turn begins with user messages
            for user_message in turn['user_messages']:
                prompt += clean_text(user_message) + \
                    self.generation_pipeline.tokenizer.eos_token
            for bot_message in turn['bot_messages']:
                prompt += clean_text(bot_message) + \
                    self.generation_pipeline.tokenizer.eos_token

        # Generate bot messages
        bot_messages = generate_responses(
            prompt,
            self.generation_pipeline,
            seed=self.seed,
            debug=self.debug,
            **self.generator_kwargs
        )
        if len(bot_messages) == 1:
            bot_message = bot_messages[0]
        else:
            bot_message = pick_best_response(
                prompt,
                bot_messages,
                self.ranker_dict,
                debug=self.debug
            )
        turn['bot_messages'].append(bot_message)
        logger.debug(f"Bot: {bot_message}")
        return bot_message

    def run(self):
        """Run the chatbot."""
        logger.info("Running the discord bot...")
        client = MyClient(self)
        client.run(self.discord.get("token"))


def run(**kwargs):
    """Run `TelegramBot`."""
    DiscordBot(**kwargs).run()