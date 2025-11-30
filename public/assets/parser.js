import console from 'console';
import * as fs from 'fs';
import * as path from 'path';
import { Parser } from './models/parser';

const fileReadOptions = {
  encoding: 'utf-8',
};

class ParserImpl implements Parser {
  private fileContent: string;

  constructor(private filePath: string) {
    this.fileContent = fs.readFileSync(this.filePath, fileReadOptions);
  }

  public parse(): void {
    const lines = this.fileContent.split('\n');
    console.log(`Lines: ${lines.length}`);
    console.log(`First line: ${lines[0]}`);
    console.log(`Last line: ${lines[lines.length - 1]}`);
  }
}

export { ParserImpl };