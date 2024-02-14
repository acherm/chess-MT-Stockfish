// read

import * as fs from 'fs';
import * as path from 'path';
import { isImpossibleCheck, setupPosition } from 'chessops/variant.js';
import { RULES } from 'chessops';
import {parseFen} from "chessops/fen.js";

function extractFen() {
    const filePath = path.join('C:/Users/axel/Git/MT_ChessEngines', 'posiciones.txt');
    const fileContents = fs.readFileSync(filePath, 'utf8');
    const positions = fileContents.split('\n').filter(Boolean);
    return positions;
}

// To use the function:
const positions = extractFen();
// console.log(positions);


const validPositions = []; // list of valid positions



// iterate over first 10 positions
for (let i = 0; i < positions.length; i++) {
    const originalFen  = positions[i].trim();

    // actually positions are non valid, since the turn (white or black) is not specified
    //const fen = [originalFen, 'w', '- - 0 1'].join(' '); // as in https://github.com/lichess-org/lila/blob/5bc21da66fa2d127ab9ea9d9a6f96526d89f8d10/ui/learn/src/ground.ts#L174
    const fen = [originalFen, 'w'].join(' ');
    const fenPos = parseFen(fen).chain(setup => setupPosition("chess", setup));
    if (fenPos.isErr) {

    }
    else {
        if (isImpossibleCheck(fenPos.unwrap())) {
            // console.log("impossible check");
        }
        else {
            validPositions.push(fen);
        }
    }


    // same with 'b' instead of 'w'
    const fen2 = [originalFen, 'b'].join(' ');
    const fenPos2 = parseFen(fen2).chain(setup => setupPosition("chess", setup));
    if (fenPos2.isErr) {

    }

    else {
        // console.log("FEN position is valid");
        // add to a list of valid positions
        if (isImpossibleCheck(fenPos2.unwrap())) {
            //  console.log("impossible check");
        }
        else {
            validPositions.push(fen2);
        }
    }


}

console.log("# valid positions: ", validPositions.length);
console.log("out of", positions.length*2);

// serialize valid positions to file
const filePath = path.join('C:/Users/axel/Git/MT_ChessEngines', 'valid_positions.txt');
const fileContents = validPositions.join('\n');

fs.writeFile(filePath, fileContents, (err) => {
    if (err) throw err;
    console.log('The file has been saved!');
});


const fenPosCase = parseFen("4B1B1/3R4/2p2R2/7K/8/r3b3/P4Q2/4k2R b").chain(setup => setupPosition("chess", setup));
if (fenPosCase.isErr) {
    console.log("FEN position is non valid....");

}
else {
    console.log("FEN position is valid");
    if (isImpossibleCheck(fenPosCase.unwrap())) {
        console.log("but impossible check!");
    }

}