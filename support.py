########################### support.py #############################
"""
This file contains a few support functions used by the other files
"""

import bots


def determine_bot_functions(bot_names):
    bot_list = []
    for name in bot_names:
        if name == "user":
            bot_list.append(bots.UserBot())
        elif name == "random":
            bot_list.append(bots.RandBot())
        elif name == "wall":
            bot_list.append(bots.WallBot())
        else:
            raise ValueError(
                """
                Bot name %s is not supported. Value names include "user",
                "random", "wall"
                """
                % name
            )
    return bot_list


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Player action timed out.")
