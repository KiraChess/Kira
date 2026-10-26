# Score and feature conventions

- Scores are integer centipawns from the side-to-move's view. Positive scores favor the side to move.
- Results are 0 for a loss, 0.5 for a draw, and 1 for a win from that same view.
- Square a1 is 0; files increase first and ranks increase by 8.
- Black's feature perspective flips ranks vertically. Pieces are encoded as friendly or enemy relative to the perspective.
- The feature index is `relative_color * 384 + (piece_kind - 1) * 64 + viewed_square`, with friendly color 0 and enemy color 1. Engine piece-kind order is pawn, knight, rook, bishop, queen, king. The exporter maps the trainer's bishop/rook order to that engine order.
- Both perspectives share the feature weights. Accumulator storage is White perspective first, Black perspective second; network output concatenation is side-to-move first.
- Float inference uses clipped ReLU in [0, 1], a fixed reduction order, output scale 400, and round-to-nearest with ties away from zero. The fixed tree reduction makes full and incremental accumulation comparable bit-for-bit.
- Accumulator state and undo tokens are bound to the network and board states that created them.
- UCI score output reports the root side-to-move's score.
- Training records stay side-to-move relative. Any White-relative interchange format requires a sign/result conversion at that boundary.
