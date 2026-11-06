package com.agriliteoffline

import android.app.ActivityManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.os.Build
import android.util.Base64
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.ln
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

class LeafInferenceModule(private val context: ReactApplicationContext) :
  ReactContextBaseJavaModule(context) {
  private val executor = Executors.newSingleThreadExecutor()
  private var interpreter: Interpreter? = null
  private var classCount = 0

  override fun getName() = "LeafInference"

  @ReactMethod
  fun deviceInfo(promise: Promise) {
    val manager = context.getSystemService(ActivityManager::class.java)
    val memory = ActivityManager.MemoryInfo()
    manager.getMemoryInfo(memory)
    promise.resolve(Arguments.createMap().apply {
      putBoolean("supported", Build.VERSION.SDK_INT >= 26 &&
        Build.SUPPORTED_ABIS.any { it == "arm64-v8a" || it == "x86_64" })
      putDouble("ramGB", memory.totalMem.toDouble() / (1024 * 1024 * 1024))
      putInt("threads", Runtime.getRuntime().availableProcessors())
      putInt("androidVersion", Build.VERSION.SDK_INT)
    })
  }

  private fun work(promise: Promise, action: () -> Unit) {
    executor.execute {
      try {
        action()
      } catch (error: Exception) {
        Log.e(name, "Offline classifier operation failed", error)
        promise.reject("LEAF_INFERENCE", error.message ?: "Leaf analysis failed", error)
      }
    }
  }

  @ReactMethod
  fun loadModel(path: String, expectedClasses: Int, promise: Promise) = work(promise) {
    val file = File(path).canonicalFile
    val directory = File(context.filesDir, "offline-models").canonicalFile
    require(file.parentFile == directory && file.isFile) { "Verified classifier file is missing." }
    require(expectedClasses == 26) { "Unsupported classifier label count." }
    interpreter?.close()
    interpreter = null
    val candidate = Interpreter(file, Interpreter.Options().apply {
      setNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
      setUseXNNPACK(true)
    })
    try {
      val input = candidate.getInputTensor(0)
      val output = candidate.getOutputTensor(0)
      require(candidate.inputTensorCount == 1 && candidate.outputTensorCount == 1 &&
        input.shape().contentEquals(intArrayOf(1, 224, 224, 3)) &&
        output.shape().contentEquals(intArrayOf(1, expectedClasses)) &&
        input.dataType() == DataType.FLOAT32 && output.dataType() == DataType.FLOAT32
      ) { "Model tensors do not match the verified mobile catalog." }
      candidate.run(ByteBuffer.allocateDirect(224 * 224 * 3 * 4).order(ByteOrder.nativeOrder()),
        Array(1) { FloatArray(expectedClasses) })
      interpreter = candidate
      classCount = expectedClasses
    } catch (error: Exception) {
      candidate.close()
      throw error
    }
    promise.resolve(null)
  }

  private fun decode(path: String): Bitmap {
    val file = File(path.removePrefix("file://")).canonicalFile
    require(file.path.startsWith(context.cacheDir.canonicalPath + File.separator)) {
      "Captured image must be in the app's temporary cache."
    }
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.path, bounds)
    require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Cannot decode this image." }
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 1600) sample *= 2
    val bitmap = requireNotNull(BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply {
      inSampleSize = sample
      inPreferredConfig = Bitmap.Config.ARGB_8888
    })) { "Cannot decode this image." }
    val orientation = ExifInterface(file.path).getAttributeInt(
      ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    val transform = Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> transform.setScale(-1f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> transform.setRotate(180f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> transform.setScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> { transform.setRotate(90f); transform.postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_90 -> transform.setRotate(90f)
      ExifInterface.ORIENTATION_TRANSVERSE -> { transform.setRotate(-90f); transform.postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_270 -> transform.setRotate(-90f)
    }
    if (transform.isIdentity) return bitmap
    val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, transform, true)
    if (rotated !== bitmap) bitmap.recycle()
    return rotated
  }

  private fun input(bitmap: Bitmap): ByteBuffer {
    val pixels = IntArray(224 * 224)
    bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
    val buffer = ByteBuffer.allocateDirect(pixels.size * 3 * 4).order(ByteOrder.nativeOrder())
    val histogram = IntArray(256)
    val gray = IntArray(pixels.size)
    val hsv = FloatArray(3)
    var plant = 0
    for ((index, pixel) in pixels.withIndex()) {
      val red = Color.red(pixel)
      val green = Color.green(pixel)
      val blue = Color.blue(pixel)
      buffer.putFloat(red / 255f)
      buffer.putFloat(green / 255f)
      buffer.putFloat(blue / 255f)
      gray[index] = (red * 0.299 + green * 0.587 + blue * 0.114).toInt().coerceIn(0, 255)
      histogram[gray[index]]++
      Color.colorToHSV(pixel, hsv)
      if ((hsv[0] in 50f..190f && hsv[1] >= 0.06f && hsv[2] >= 0.04f) ||
        (hsv[0] in 10f..50f && hsv[1] >= 0.12f && hsv[2] in 0.12f..0.79f)) plant++
    }
    val entropy = -histogram.filter { it > 0 }.sumOf {
      val probability = it.toDouble() / pixels.size
      probability * ln(probability) / ln(2.0)
    }
    var gradient = 0.0
    for (y in 1 until 224) for (x in 1 until 224) {
      val index = y * 224 + x
      gradient += abs(gray[index] - gray[index - 1]) + abs(gray[index] - gray[index - 224])
    }
    require(plant.toDouble() / pixels.size >= 0.15) {
      "Confidence that this is a usable leaf image is too low: too little leaf-like color. Please capture or select a clear image of a real tomato or chilli leaf in natural light."
    }
    require(entropy >= 4.0 && gradient / (223 * 223 * 2) >= 1.5) {
      "Confidence that this is a usable leaf image is too low: the image is too uniform or blurry. Please capture or select a clear image of a real tomato or chilli leaf in natural light."
    }
    buffer.rewind()
    return buffer
  }

  @ReactMethod
  fun predict(path: String, promise: Promise) = work(promise) {
    val model = requireNotNull(interpreter) { "Load the classifier before taking a photo." }
    val bitmap = decode(path)
    try {
      val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
      val scores = Array(1) { FloatArray(classCount) }
      try {
        model.run(input(resized), scores)
      } finally {
        if (resized !== bitmap) resized.recycle()
      }
      val scale = minOf(1f, 512f / maxOf(bitmap.width, bitmap.height))
      val preview = Bitmap.createScaledBitmap(bitmap,
        (bitmap.width * scale).toInt().coerceAtLeast(1),
        (bitmap.height * scale).toInt().coerceAtLeast(1), true)
      val bytes = ByteArrayOutputStream()
      try {
        check(preview.compress(Bitmap.CompressFormat.JPEG, 75, bytes)) { "Cannot create image preview." }
      } finally {
        if (preview !== bitmap) preview.recycle()
      }
      promise.resolve(Arguments.createMap().apply {
        putArray("scores", Arguments.createArray().apply {
          scores[0].forEach { pushDouble(it.toDouble()) }
        })
        putString("photo", "data:image/jpeg;base64," + Base64.encodeToString(bytes.toByteArray(), Base64.NO_WRAP))
      })
    } finally {
      bitmap.recycle()
    }
  }

  @ReactMethod
  fun unload(promise: Promise) = work(promise) {
    interpreter?.close()
    interpreter = null
    classCount = 0
    promise.resolve(null)
  }

  override fun invalidate() {
    executor.execute { interpreter?.close(); interpreter = null }
    executor.shutdown()
    super.invalidate()
  }
}
