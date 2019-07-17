import random
from collections import Counter


class List:
    def __init__(list, player, next=None):
        list.player = player
        list.next = next
        list.cards = Counter()

    def __iter__(list):
        yield list
        if list.next is not None:
            yield from list.next

    def __str__(list):
        return str(list.player)

player_list = List("wesley", List("david", List("esther", List("clare"))))
player_list.next.next.next.next = player_list

def play_game():
    player_list = List("wesley", List("david", List("esther", List("clare"))))
    player_list.next.next.next.next = player_list
    for player in player_list:
        new_card = random.choice(["exploding kitten", "diffuse"])
        print("%s picks up %s!" % (player, new_card))
        if new_card == "exploding kitten":
            if "diffuse" in player.cards:
                player.cards["diffuse"] -= 1
                print("%s plays a %s card" % (player, "diffuse"))
            else:
                print("%s explodes!!!!" % player)
                break
        else:
            player.cards[new_card] += 1
