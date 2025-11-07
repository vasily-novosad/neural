import path from 'node:path';
import NodemonPlugin from 'nodemon-webpack-plugin';
import type { Configuration } from 'webpack';
import webpack from 'webpack';

import packageInfo from '../package.json' with { type: 'json' };
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isDev = process.env.NODE_ENV === 'development';
const webpackBaseConfig: Configuration = {
  target: 'node',
  mode: isDev ? 'development' : 'production',
  devtool: isDev ? 'source-map' : false,
  optimization: {
    minimize: !isDev,
  },
  entry: {
    index: path.resolve(__dirname, '../src/index.ts'),
  },
  output: {
    libraryTarget: 'commonjs2',
    path: isDev ? path.join(__dirname, '../build/') : path.join(__dirname, '../dist/'),
    filename: '[name].js',
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js', '.json'],
    alias: {
      '~': path.resolve(__dirname, '..', 'src'),
    },
  },
  externals: [
    /^@via-profit-services\/geography\/data\/.*/,
    // KNEX ignore not postgress drivers
    /sqlite3/,
    /mysql2/,
    /mssql/,
    /mariasql/,
    /mysql/,
    /oracle/,
    /strong-oracle/,
    /oracledb/,
    /pg-native/,
    /pg-query-stream/,
    /import-file/,
  ],
  node: {
    __dirname: true,
  },
  plugins: [
    /**
     * Development and production plugins
     */
    new webpack.DefinePlugin({
      'process.env.WEBPACK_INJECT_APP_VERSION': JSON.stringify(packageInfo.version),
    }),
    ...(isDev
      ? /**
         * Development only plugins
         */
        [
          new NodemonPlugin({
            watch: ['./build'],
            exec: 'node --inspect=9229 ./build/index.js',
          }),
        ]
      : /**
         * Production only plugins
         */
        []),
  ],
};

export default webpackBaseConfig;
