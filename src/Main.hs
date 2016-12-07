import Control.Monad ( void )
import qualified OpenCV as CV
import qualified Data.ByteString as B
import System.Environment


main :: IO ()
main = do
  [file] <- getArgs

  img <- CV.imdecode CV.ImreadUnchanged <$> B.readFile file
  CV.withWindow "test" $ \window -> do
    CV.imshow window img
    void $ CV.waitKey 0
